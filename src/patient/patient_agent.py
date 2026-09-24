"""
Patient agent node for LangGraph.
"""
import logging
from typing import Dict, Any, List, Type

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel

from ..utils import get_llm_model
from ..utils.constants import PATIENT_PREFIX, THERAPIST_PREFIX
from ..utils.formatting import (
    format_speaker_utterance,
    transform_messages_for_patient,
)
from ..utils.tracing import enrich_span, traced_span
from ..utils.store import PATIENT_MEMORY_NS
from . import patient_config
from .patient_dtos import (
    PatientProfileData,
    PatientLatentVariables,
    PatientCognitiveDirective
)
from ..graphs.graph_config import LatentTrajectoryEntry
from .patient_planner_prompts import build_patient_prompt
from ..causal_study.metrics import calculate_token_metrics
from ..utils.config import CAUSAL_RECORD_BASELINE

logger = logging.getLogger(__name__)


def _build_model_with_fallback(output_schema: Type[BaseModel], temperature: float, include_raw: bool = False):
    """Build a structured-output model chain (primary + fallback) for the given schema."""
    provider_prefs = patient_config._config.get_provider_preferences()
    logprobs = True if CAUSAL_RECORD_BASELINE else None
    top_logprobs = 1 if CAUSAL_RECORD_BASELINE else None

    primary = (
        get_llm_model(
            temperature=temperature,
            provider=patient_config._config.PROVIDER,
            model_name=patient_config._config.MODEL_NAME,
            provider_preferences=provider_prefs,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        .with_structured_output(output_schema, method="json_mode", include_raw=include_raw)
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=patient_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    fallback = (
        get_llm_model(
            temperature=temperature,
            provider=patient_config._config.PROVIDER,
            model_name=patient_config._config.FALLBACK_MODEL_NAME,
            provider_preferences=provider_prefs,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        .with_structured_output(output_schema, method="json_mode", include_raw=include_raw)
        .with_retry(
            retry_if_exception_type=(Exception,),
            stop_after_attempt=patient_config.API_MAX_RETRIES,
            wait_exponential_jitter=True,
        )
    )
    return primary.with_fallbacks([fallback])


@traced_span("patient")
def patient_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    """
    Patient node in the therapy conversation graph.

    Runs a single LLM call per turn using the rolling memory read from the 
    LangGraph Store via runtime.store.
    """
    try:
        logger.info("patient_node invoked")

        configurable = config.get("configurable", {})

        current_session_number = configurable.get("current_session_number")
        if current_session_number is None:
            raise ValueError("current_session_number missing from configurable.")

        patient_profile_data_dict = configurable.get("patient_profile_data")
        if not patient_profile_data_dict:
            raise ValueError("patient_profile_data missing or empty in configurable.")

        # Calculate patient's current turn in the session (roughly half of turn_count)
        # Assuming one therapist message, then one patient message.
        # It's safer to just count the patient's own turns.
        all_messages = state.get("messages")
        if not all_messages:
            raise ValueError("state['messages'] is missing or empty; no therapist turn to respond to.")
        
        patient_turns = sum(1 for m in all_messages if m.name == "Patient") + 1
        turn_number = state.get("turn_count", 0) + 1
        
        enrich_span(
            node_name="patient",
            description=f"Patient responds in session {current_session_number}, turn {turn_number}",
            session_number=current_session_number,
            turn_number=turn_number,
            speaker="patient",
            phase="conversation",
            purpose="generate_response",
            model_provider=patient_config._config.PROVIDER,
            model_name=patient_config._config.MODEL_NAME,
        )

        temperature = patient_config._config.get_temperature()
        current_turns = state.get("current_session_turns", [])

        latent_vars = state.get("patient_latent_variables")
        if not latent_vars:
            raise ValueError("patient_latent_variables missing or empty in state.")

        for key in ("self_efficacy", "anger", "problem_recognition", "motivational_readiness"):
            if latent_vars.get(key) is None:
                raise ValueError(f"patient_latent_variables['{key}'] is missing.")

        self_efficacy_score = float(latent_vars["self_efficacy"])
        anger_score = float(latent_vars["anger"])
        problem_recognition_score = float(latent_vars["problem_recognition"])
        motivational_readiness_score = float(latent_vars["motivational_readiness"])
        latent_vars_model = PatientLatentVariables.model_validate(latent_vars)

        from ..utils.config import DISABLE_COGNITIVE_INTERPRETER_IMPACT, DISABLE_PATIENT_MEMORY

        # --- Read rolling memory from store ---
        store = runtime.store
        bse_item = store.get(PATIENT_MEMORY_NS, "between_session_occurrences")
        between_session_occurrences: List[str] = bse_item.value.get("items", []) if bse_item else []

        if DISABLE_PATIENT_MEMORY:
            logger.info("Ablation Mode active: DISABLE_PATIENT_MEMORY (patient rolling memory omitted from prompt inputs, using summary memory)")
            summary_item = store.get(PATIENT_MEMORY_NS, "patient_summary_memory")
            previous_session_summary = summary_item.value.get("summary", "") if summary_item and summary_item.value else ""
            facts: List[str] = []
            agreed_change_plans: List[Dict[str, Any]] = []
            implicit_threads: List[str] = []
            motivations: List[str] = []
            concerns: List[str] = []
            persona_and_stylistic_baseline: str = ""
        else:
            previous_session_summary = ""
            facts_item = store.get(PATIENT_MEMORY_NS, "facts")
            plans_item = store.get(PATIENT_MEMORY_NS, "agreed_change_plans")
            threads_item = store.get(PATIENT_MEMORY_NS, "implicit_threads")
            
            motivations_item = store.get(PATIENT_MEMORY_NS, "motivations")
            concerns_item = store.get(PATIENT_MEMORY_NS, "concerns")
            persona_item = store.get(PATIENT_MEMORY_NS, "persona_and_stylistic_baseline")

            facts: List[str] = facts_item.value.get("items", []) if facts_item else []
            agreed_change_plans: List[Dict[str, Any]] = plans_item.value.get("items", []) if plans_item else []
            implicit_threads: List[str] = threads_item.value.get("items", []) if threads_item else []
            motivations: List[str] = motivations_item.value.get("items", []) if motivations_item else []
            concerns: List[str] = concerns_item.value.get("items", []) if concerns_item else []
            persona_and_stylistic_baseline: str = persona_item.value.get("value", "") if persona_item else ""

        dynamic_directives = state.get("patient_state_reasoning", {})

        if DISABLE_COGNITIVE_INTERPRETER_IMPACT:
            logger.info("Ablation Mode active: DISABLE_COGNITIVE_INTERPRETER_IMPACT (cognitive interpreter bypassed, state & cognitive shifts omitted from patient prompt)")

        profile_data = PatientProfileData.model_validate(patient_profile_data_dict)

        patient_prompt = build_patient_prompt(
            profile_data=profile_data,
            anger_score=anger_score,
            self_efficacy_score=self_efficacy_score,
            problem_recognition_score=problem_recognition_score,
            motivational_readiness_score=motivational_readiness_score,
            facts=facts,
            agreed_change_plans=agreed_change_plans,
            implicit_threads=implicit_threads,
            motivations=motivations,
            concerns=concerns,
            persona_and_stylistic_baseline=persona_and_stylistic_baseline,
            dynamic_directives=dynamic_directives,
            disable_cognitive_interpreter_impact=DISABLE_COGNITIVE_INTERPRETER_IMPACT,
            disable_patient_memory=DISABLE_PATIENT_MEMORY,
            previous_session_summary=previous_session_summary,
        )
        
        from .patient_config import PATIENT_CONTEXT_WINDOW_TURNS

        # Limit to the last N turns (1 turn = 2 messages)
        max_messages = PATIENT_CONTEXT_WINDOW_TURNS * 2
        recent_messages = all_messages[-max_messages:] if len(all_messages) > max_messages else all_messages

        history_text = ""
        for msg in recent_messages[:-1]:
            prefix = PATIENT_PREFIX if msg.name == "Patient" else THERAPIST_PREFIX
            history_text += f"{prefix}{msg.content}\n"
            
        last_therapist_msg = recent_messages[-1]
        
        agent_messages = [
            SystemMessage(content=patient_prompt),
        ]
        
        # BSE mandatory topic generator
        if current_session_number > 1 and between_session_occurrences and patient_turns <= 3:
            most_recent_occurrence = between_session_occurrences[-1]
            logger.info(f"BSE_Directive_Active: turn {patient_turns}")
            bse_rule = (
                "<instructions>\n"
                "We are at the beginning of a new session. Please organically bring up the recent events "
                "that occurred since your last meeting: "
                f"{most_recent_occurrence}\n"
                "Over the next few turns, explore these events collaboratively with the therapist. "
                "Weave these events into your `pressing_disclosure_intention`.\n"
                "</instructions>"
            )
            agent_messages.append(SystemMessage(content=bse_rule))
            
        user_content = f"<conversation_history>\n{history_text.strip()}\n</conversation_history>\n\n<latest_therapist_turn>\n{THERAPIST_PREFIX}{last_therapist_msg.content}\n</latest_therapist_turn>"
        agent_messages.append(HumanMessage(content=user_content))
        
        include_raw = CAUSAL_RECORD_BASELINE
        agent_model = _build_model_with_fallback(PatientCognitiveDirective, temperature, include_raw=include_raw)
        invoke_result = agent_model.invoke(agent_messages, config=config)

        raw_msg = None
        if include_raw and isinstance(invoke_result, dict) and "parsed" in invoke_result:
            agent_result: PatientCognitiveDirective = invoke_result["parsed"]
            raw_msg = invoke_result.get("raw")
        else:
            agent_result = invoke_result

        if not getattr(agent_result, "response", None):
            raise ValueError(f"Agent generated an incomplete directive: {agent_result}")

        token_metrics = None
        if raw_msg is not None:
            logprobs_data = None
            if hasattr(raw_msg, "response_metadata") and isinstance(raw_msg.response_metadata, dict):
                logprobs_data = raw_msg.response_metadata.get("logprobs")
            if not logprobs_data and hasattr(raw_msg, "additional_kwargs") and isinstance(raw_msg.additional_kwargs, dict):
                logprobs_data = raw_msg.additional_kwargs.get("logprobs")

            content_logprobs = None
            if isinstance(logprobs_data, dict):
                content_logprobs = logprobs_data.get("content")
            elif hasattr(logprobs_data, "content"):
                content_logprobs = logprobs_data.content

            if content_logprobs:
                token_metrics = calculate_token_metrics(content_logprobs)
            elif CAUSAL_RECORD_BASELINE:
                logger.warning(
                    f"CAUSAL_RECORD_BASELINE is true but no logprobs returned in raw response metadata: {logprobs_data}. "
                    "Ensure OPENROUTER_LOGPROB_PROVIDERS points to a provider supporting logprobs (e.g. Parasail)."
                )

        logger.info(
            "Patient Directive generated",
            extra={
                "conversational_stance": agent_result.conversational_stance,
                "optimizing_for": agent_result.cognitive_layer.optimizing_for,
                "defense_pattern": agent_result.psychological_layer.defense_pattern
            }
        )

        response_content = agent_result.response
        patient_message = AIMessage(content=response_content, name="Patient")
        summary_line = format_speaker_utterance(response_content, PATIENT_PREFIX)
        logger.info(summary_line)

        if motivational_readiness_score <= 33.3:
            current_dominant_stage = "Precontemplation"
        elif motivational_readiness_score <= 66.6:
            current_dominant_stage = "Contemplation"
        else:
            current_dominant_stage = "Action"

        patient_turn_num = len(current_turns) + 1
        current_lvs = {
            "anger": anger_score,
            "self_efficacy": self_efficacy_score,
            "problem_recognition": problem_recognition_score,
            "motivational_readiness": motivational_readiness_score,
        }

        turn_record = {
            "turn_number": patient_turn_num,
            "speaker": "patient",
            "volley": response_content,
            "current_dominant_stage": current_dominant_stage,
            "patient_latent_variables": current_lvs,
        }

        if CAUSAL_RECORD_BASELINE:
            turn_record["package"] = dynamic_directives
            turn_record["therapist_turn"] = last_therapist_msg.content
            turn_record["history"] = history_text.strip()
            turn_record["baseline_system_prompt"] = patient_prompt
            if current_session_number > 1 and between_session_occurrences and patient_turns <= 3:
                turn_record["bse_occurrence"] = between_session_occurrences[-1]
            if token_metrics:
                turn_record["token_metrics"] = token_metrics

        enrich_span(
            current_dominant_stage=current_dominant_stage,
            token_perplexity=token_metrics.get("perplexity") if token_metrics else None,
            avg_logprob=token_metrics.get("avg_logprob") if token_metrics else None,
            total_logprob=token_metrics.get("total_logprob") if token_metrics else None,
        )

        update_dict = {
            "messages": state["messages"] + [patient_message],
            "current_speaker": "patient",
            "turn_count": state.get("turn_count", 0) + 1,
            "current_session_turns": current_turns + [turn_record],
        }

        from ..utils.config import (
            ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES,
            DISABLE_COGNITIVE_INTERPRETER_IMPACT,
            VANILLA_BASELINE,
            CONSISTENT_CLIENT_BASELINE,
            SIMPATIENT_BASELINE,
            PATIENT_PSI_BASELINE,
        )
        ci_active = not (
            ABLATE_PATIENT_COGNITION_STATIC_HIDDEN_VARIABLES
            or DISABLE_COGNITIVE_INTERPRETER_IMPACT
            or VANILLA_BASELINE
            or CONSISTENT_CLIENT_BASELINE
            or SIMPATIENT_BASELINE
            or PATIENT_PSI_BASELINE
        )

        # Snapshot for all_latent_variable_logs
        snapshot = {
            "turn_number": patient_turn_num,
            "anger": anger_score,
            "self_efficacy": self_efficacy_score,
            "problem_recognition": problem_recognition_score,
            "motivational_readiness": motivational_readiness_score,
        }
        snapshots = list(state.get("latent_variable_logs", []))
        snapshots.append(snapshot)
        update_dict["latent_variable_logs"] = snapshots

        if ci_active:
            trajectory_entry: LatentTrajectoryEntry = {
                "step_index": -1,
                "component": "patient",
                "session_number": current_session_number,
                "turn_number": patient_turn_num,
                "anger": float(anger_score),
                "self_efficacy": float(self_efficacy_score),
                "problem_recognition": float(problem_recognition_score),
                "motivational_readiness": float(motivational_readiness_score),
                "details": {
                    "current_dominant_stage": current_dominant_stage,
                },
            }
            session_trajectory = list(state.get("session_latent_trajectory", []))
            session_trajectory.append(trajectory_entry)
            update_dict["session_latent_trajectory"] = session_trajectory

        if not state.get("patient_system_prompt"):
            update_dict["patient_system_prompt"] = patient_prompt

        return update_dict

    except Exception as e:
        logger.exception(f"Patient node failed: {e}")
        raise
