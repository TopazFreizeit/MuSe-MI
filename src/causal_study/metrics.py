"""
Mathematical Token Perplexity & Log-Likelihood Calculation Utilities.

Computes Total Log-Likelihood, Average Token Logprob, and Perplexity directly
from token log probabilities returned by meta-llama/llama-3.3-70b-instruct via OpenRouter.
"""
import math
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


def calculate_token_metrics(
    logprobs_content: Optional[Sequence[Union[Any, Dict[str, Any]]]]
) -> Dict[str, Any]:
    """
    Calculate Token Perplexity, Average Logprob, and Total Log-Likelihood from token logprobs.

    Handles both OpenAI SDK ChoiceLogprobs objects (with .token and .logprob attributes)
    and deserialized dictionaries (with 'token' and 'logprob' keys).
    Filters out items where logprob is None (e.g. special role tokens or initial tokens).

    Args:
        logprobs_content: Sequence of token logprob objects or dicts from the model response.

    Returns:
        Dictionary containing:
            - token_count: Number of valid scored tokens.
            - total_logprob: Sum of log probabilities (Total Log-Likelihood).
            - avg_logprob: Mean token log probability.
            - perplexity: Exponentiated negative average logprob (exp(-avg_logprob)).
            - tokens: List of token detail dicts [{"token": str, "logprob": float}].
    """
    if not logprobs_content:
        return {
            "token_count": 0,
            "total_logprob": None,
            "avg_logprob": None,
            "perplexity": None,
            "tokens": [],
        }

    valid_logprobs: List[float] = []
    token_details: List[Dict[str, Any]] = []

    for item in logprobs_content:
        if isinstance(item, dict):
            token_str = str(item.get("token", ""))
            lp = item.get("logprob")
        else:
            token_str = str(getattr(item, "token", ""))
            lp = getattr(item, "logprob", None)

        if lp is not None:
            try:
                lp_float = float(lp)
                valid_logprobs.append(lp_float)
                token_details.append({"token": token_str, "logprob": round(lp_float, 4)})
            except (ValueError, TypeError):
                logger.debug(f"Skipping non-numeric logprob: {lp}")
        else:
            # Token with None logprob (e.g. bos/role tokens)
            token_details.append({"token": token_str, "logprob": None})

    token_count = len(valid_logprobs)
    if token_count == 0:
        return {
            "token_count": 0,
            "total_logprob": None,
            "avg_logprob": None,
            "perplexity": None,
            "tokens": token_details,
        }

    total_logprob = sum(valid_logprobs)
    avg_logprob = total_logprob / token_count
    # Perplexity = exp(- (1/N) * sum(log p(w_i)))
    perplexity = math.exp(-avg_logprob)

    return {
        "token_count": token_count,
        "total_logprob": round(total_logprob, 4),
        "avg_logprob": round(avg_logprob, 4),
        "perplexity": round(perplexity, 4),
        "tokens": token_details,
    }


def apply_package_math(prior_state: Dict[str, Any], package: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply cognitive simulator mathematical transition rules using a serialized appraisal package.

    Args:
        prior_state: Dictionary with 'anger', 'self_efficacy', 'problem_recognition', 'motivational_readiness'.
        package: Appraisal package C_t containing per-variable 'reasoning' and 'impact'.

    Returns:
        Updated latent state dictionary V_{k,t}.
    """
    from ..patient_state_manager.patient_state_manager_dtos import SemanticImpact
    from ..patient_state_manager.cognitive_simulator import DeterministicCognitiveSimulator

    class _VariableImpact:
        def __init__(self, impact_str: str):
            norm = str(impact_str or "Neutral").strip().capitalize()
            if norm == "Success":
                self.semantic_impact = SemanticImpact.SUCCESS
            elif norm == "Resistance":
                self.semantic_impact = SemanticImpact.RESISTANCE
            else:
                self.semantic_impact = SemanticImpact.NEUTRAL

    class _MockDTO:
        def __init__(self, pkg: Dict[str, Any]):
            self.anger = _VariableImpact(pkg.get("anger", {}).get("impact", "Neutral"))
            self.self_efficacy = _VariableImpact(pkg.get("self_efficacy", {}).get("impact", "Neutral"))
            self.problem_recognition = _VariableImpact(pkg.get("problem_recognition", {}).get("impact", "Neutral"))
            self.motivational_readiness = _VariableImpact(pkg.get("motivational_readiness", {}).get("impact", "Neutral"))

    return DeterministicCognitiveSimulator.apply_math(prior_state, _MockDTO(package))
