import logging
import os
import random
import json
import heapq
import numpy as np
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import JsonOutputParser
from langgraph.runtime import Runtime

from src.utils import get_llm_model
from src.utils.tracing import enrich_span, traced_span
from src.utils.formatting import format_speaker_utterance
from src.utils.constants import PATIENT_PREFIX, THERAPIST_PREFIX
from src.patient.patient_dtos import PatientProfileData
from src.patient.patient_config import PATIENT_CONTEXT_WINDOW_TURNS, PatientConfigClass
from src.utils.store import PATIENT_MEMORY_NS
from .utils import topic_graph, topic2description

logger = logging.getLogger(__name__)

_retriever_cache = {}

def get_retriever():
    if "tokenizer" not in _retriever_cache:
        logger.info("Loading BAAI/bge-reranker-v2-m3...")
        retriever_path = os.getenv("RETRIEVER_PATH", "BAAI/bge-reranker-v2-m3")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        _retriever_cache["tokenizer"] = AutoTokenizer.from_pretrained(retriever_path)
        _retriever_cache["model"] = AutoModelForSequenceClassification.from_pretrained(retriever_path).to(device)
        _retriever_cache["model"].eval()
        _retriever_cache["device"] = device
    return _retriever_cache["tokenizer"], _retriever_cache["model"], _retriever_cache["device"]



class ConsistentClientAdapter:
    def __init__(self, profile_data: PatientProfileData, c_state_dict: dict, memory_prompt: str = ""):
        self.goal = profile_data.topic
        self.behavior = profile_data.behavior
        self.personas = c_state_dict.get("remaining_personas", profile_data.personas.copy())
        
        self.motivation = profile_data.motivation[-1]
        self.engagemented_topics = profile_data.motivation[:-1]

        self.beliefs = c_state_dict.get("remaining_beliefs", profile_data.beliefs.copy())
        self.acceptable_plans = c_state_dict.get("remaining_plans", profile_data.acceptable_plans.copy())
        
        self.initial_stage = profile_data.initial_state
        self.receptivity = int(sum(profile_data.suggestibilities) / len(profile_data.suggestibilities))
        self.memory_prompt = memory_prompt

        self.action2prompt = {
            "Deny": "You should directly refuse to admit your behavior is problematic or needs change.",
            "Downplay": "You should downplay the importance or impact of your behavior.",
            "Blame": "You should blame external factors or others to justify your behavior.",
            "Inform": "You should share details about your background, experiences, or emotions revealing the current state.",
            "Engage": "You should interact with counselor consistently based on your state and mimic the style in the reference conversation.",
            "Hesitate": "You should show uncertainty, indicating ambivalence about change.",
            "Doubt": "You should expresse skepticism about the practicality or success of proposed changes but not reveal further information.",
            "Acknowledge": "You should acknowledge the need for change.",
            "Accept": "You should agree to adopt the suggested action plan.",
            "Reject": "You should decline the proposed plan, deeming it unsuitable.",
            "Plan": "You should propose or detail steps for a change plan.",
            "Terminate": "You should highlight current state and engagement, express a desire to end the current session, and suggest further discussion be deferred to a later time.",
        }

        self.state2prompt = {
            "Precontemplation": f"You doesn't think your {self.behavior} is problematic and wants to sustain.",
            "Contemplation": f"You feels that your {self.behavior} is problematic, but still hesitate about {self.goal}.",
            "Preparation": f"You gets ready to take action to change and begins discuss about steps toward {self.goal}.",
        }
        
        self.topic2description = topic2description(self.behavior, self.goal)
        self.topic_graph = topic_graph

    def _call_llm(self, messages, temperature=0.7):
        # Build LLM
        config = PatientConfigClass()
        model = get_llm_model(
            temperature=temperature,
            provider=config.PROVIDER,
            model_name=config.MODEL_NAME,
            provider_preferences=config.OPENROUTER_PROVIDER_PREFS
        )
        
        fallback = get_llm_model(
            temperature=temperature,
            provider=config.PROVIDER,
            model_name=config.FALLBACK_MODEL_NAME,
            provider_preferences=config.OPENROUTER_PROVIDER_PREFS
        )
        
        chain = model.with_fallbacks([fallback])
        response = chain.invoke(messages)
        content = response.content
        
        return content

    def get_all_topics(self):
        all_topics = []
        for nodes in self.topic_graph:
            if nodes not in all_topics:
                all_topics.append(nodes)
            for node in self.topic_graph[nodes]:
                if node not in all_topics:
                    all_topics.append(node)
        
        passages = []
        topic2desc = topic2description(self.behavior, self.goal)
        for topic in all_topics:
            passages.append(topic2desc.get(topic, ""))
        return all_topics, passages

    def top5_related_topics(self, context):
        all_topics, passages = self.get_all_topics()
        query = context[-1].split("Therapist: ")[-1] if "Therapist: " in context[-1] else context[-1]
        queries = [query] * len(all_topics)
        query_evids = list(zip(queries, passages))
        
        tokenizer, model, device = get_retriever()
        with torch.no_grad():
            inputs = tokenizer(
                query_evids,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_scores = model(**inputs, return_dict=True).logits.view(-1,).float()
            scores = torch.sigmoid(batch_scores).tolist()
            
        top_5_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        return [all_topics[idx] for idx in top_5_indices]

    def verify_motivation(self, context):
        prompt = """Your task is to evaluate whether the Counselor's responses align with the Client's motivation concerning a specific topic, target (self or others), and aspect (risk or benefit). Determine if the Counselor's statements effectively motivates the Client. Your analysis should be logical, thorough, and well-supported, providing clear analysis at each step.

Here are some examples to help you understand the task better:
## Example 1:
### Input
Here is the conversation snippet toward reducing alcohol consumption:
- Counselor: Hello. How are you?
- Client: I am good. What about you?
- Counselor: I'm doing well, thank you. I understand you wanted to talk about your alcohol consumption. Can you share a bit more about how you're feeling about it?

The Motivation of Client is as follows:
- You are motivated because of the risk of drinking alcohol in relation to depression for yourself, as alcohol could worsen your depression.

Question: Can the Counselor's statement motivate the Client?

### Output
Analysis: The Counselor's initial statement focuses on building rapport and asking the Client to share their feelings about alcohol consumption, but it does not directly address the Client's specific motivation—the risk of alcohol exacerbating depression. Since the Client is motivated by the personal risk of worsening depression, an effective motivational approach would involve acknowledging that risk and connecting it to the Client's emotional or mental health concerns. The Counselor’s statement lacks any mention of the risks or the Client's depression, making it less likely to effectively motivate the Client in this context.
Answer: No


## Example 2:
### Input
Here is the conversation snippet toward reducing alcohol consumption:
- Counselor: Are you surprised what that might be true?
- Client: Yeah, and a couple of my friends drink too.
- Counselor: Well, you might not be drinking that much, and other kids are also trying alcohol. I'd like to share with you the risk of using. Alcohol and drugs could really harm you because your brain is still changing. It also-- you're very high risk for becoming addicted. Alcohol and drugs could also interfere with your role in life and your goals, especially in sports, and it could cause unintended sex. How do you feel about this information?

The Motivation of Client is as follows:
- You are motivated because of the risk of drinking alcohol in sports for yourself, as alcohol would affect your ability to play soccer.

Question: Can the Counselor's statement motivate the Client?

### Output
Analysis: The Counselor's statement addresses various risks associated with alcohol use, including its potential impact on the Client’s role in life and goals, particularly in sports. Since the Client's motivation revolves around the risk of alcohol affecting their ability to play soccer, the Counselor’s mention of how alcohol could interfere with sports aligns with the Client's concern. By highlighting this specific risk, the Counselor's statement effectively taps into the Client’s personal motivation, making it more likely to encourage behavior change.
Answer: Yes


## Example 3:
### Input
Here is the conversation snippet toward reducing alcohol consumption:
- Counselor: It sounds like you're considering making some changes around your alcohol consumption. What makes you think it might be good to cut back?
- Client: I guess I just want to be more mindful of my health and well-being.
- Counselor: It sounds like you're considering making some positive changes for your health. What are some reasons that are motivating you to cut back on alcohol?

The Motivation of Client is as follows:
- You are motivated because of the risk of drinking alcohol in relation to depression for yourself, as alcohol could worsen your depression.

Question: Can the Counselor's statement motivate the Client?

### Output
Analysis: While the Counselor’s statement touches on the Client’s general motivation to improve their health, it does not specifically address the Client's key motivation—the risk of alcohol worsening their depression. The Counselor asks broad questions about the Client’s reasons for cutting back on alcohol, but fails to connect directly to the Client’s concern about depression, which is a central aspect of their motivation. For the statement to effectively motivate the Client, it would need to focus more on the specific risk of alcohol impacting their mental health.
Answer: No

## Example 4:
### Input
Here is the conversation snippet toward reducing alcohol consumption:
- Counselor: It seems like you're concerned about staying productive at work, but drinking regularly could have some negative effects. Alcohol can make it harder to stay focused and might even cause you to miss deadlines or make mistakes. Have you noticed any of those risks affecting your productivity?
- Counselor: Not really, I still get my work done, and I don’t feel like my drinking is hurting my performance. I mean, I can still function well enough, so I don’t think it’s a problem.
- Counselor: That makes sense, but over time, regular drinking can slowly take a toll on your ability to perform at your best. You might not notice it now, but it could lead to more mistakes or slower work in the future. Are you worried that alcohol could start to interfere with your productivity in the long run?

The Motivation of Client is as follows:
- You are motivated because of the benefit of reducing alcohol consumption in terms of productivity for yourself, as you feel more productive when you don’t have a hangover.

Question: Can the Counselor's statement motivate the Client?

### Output
Analysis:The Client is motivated by the benefit of increased productivity without alcohol, but the Counselor focuses on the risk of future productivity loss from drinking. The Counselor’s focus on potential risks doesn't align with the Client's motivation, which is based on the immediate benefit of feeling more productive when avoiding alcohol. To be more effective, the Counselor should have highlighted the benefit the Client already experiences.
Answer: No

Now, Here is the conversation snippet toward [@goal]:
- [@context]

The Motivation of Client is as follows:
- [@motivation]

Question: Can the Counselor's statement motivate the Client?

#### Output
"""
        prompt = prompt.replace("[@goal]", self.goal)
        prompt = prompt.replace("[@context]", "\n- ".join(context[-5:]))
        prompt = prompt.replace("[@motivation]", self.motivation)
        
        response = self._call_llm([HumanMessage(content=prompt)], temperature=0.2)
        return response.split("\n")[0].split(": ")[-1]

    def select_action(self, context_str, state_dict):
        prompt = """Assume you are a Client involved in a counseling conversation. The current conversation is provided below:
[@context]

Based on the context, allocate probabilities to each of the following dialogue actions to maintain coherence:
- Deny: The client should directly refuse to admit their behavior is problematic or needs change without additional reasons.
- Downplay: The client should downplay the importance or impact of their behavior or situation.
- Blame: The client should blame external factors or others to justify their behavior.
- Inform: The client should share details about their background, experiences, or emotions.
- Engage: The client interacts politely with the counselor, such as greeting or thanking.

Output ONLY valid JSON. Ensure that the sum of all probabilities equals 100. For example:
```json
{"Deny": 35, "Downplay": 25, "Blame": 25, "Inform": 5, "Engage": 10}
```
"""
        prompt = prompt.replace(
            "[@context]",
            context_str.replace("Client:", "**Client**:").replace("Counselor:", "**Counselor**:"),
        )
        context_aware_action_distribution = None
        for _ in range(2):
            response = self._call_llm([HumanMessage(content=prompt)], temperature=0.2)
            # We already have JSON extraction handled inside _call_llm if method is json.
            # However, fallback parsing just in case:
            if isinstance(response, str):
                try:
                    context_aware_action_distribution = JsonOutputParser().invoke(response)
                except Exception as e:
                    logger.warning(f"Failed to parse JSON from LLM response: {response}")
                    raise e
                    # continue
            elif isinstance(response, dict):
                context_aware_action_distribution = response
            
            if context_aware_action_distribution:
                break

        if not context_aware_action_distribution:
            context_aware_action_distribution = {
                "Deny": 20,
                "Downplay": 20,
                "Blame": 20,
                "Engage": 20,
                "Inform": 20,
            }
            
        receptivity = state_dict.get("consistent_receptivity", self.receptivity)
        
        if receptivity < 2:
            receptivity_aware_action_distribution = {"Deny": 23, "Downplay": 28, "Blame": 15, "Engage": 11, "Inform": 22}
        elif receptivity < 3:
            receptivity_aware_action_distribution = {"Deny": 20, "Downplay": 25, "Blame": 10, "Engage": 15, "Inform": 30}
        elif receptivity < 4:
            receptivity_aware_action_distribution = {"Deny": 19, "Downplay": 21, "Blame": 11, "Engage": 13, "Inform": 36}
        elif receptivity < 5:
            receptivity_aware_action_distribution = {"Deny": 9, "Downplay": 20, "Blame": 13, "Engage": 14, "Inform": 44}
        else:
            receptivity_aware_action_distribution = {"Deny": 7, "Downplay": 13, "Blame": 4, "Engage": 16, "Inform": 60}
            
        action_distribution = {
            action: context_aware_action_distribution.get(action, 0) + receptivity_aware_action_distribution.get(action, 0)
            for action in receptivity_aware_action_distribution
        }
        
        if len(self.personas) == 0:
            action_distribution["Inform"] = 0
        if len(self.beliefs) == 0:
            action_distribution["Blame"] = 0
            
        total = sum(action_distribution.values())
        if total == 0:
            return "Engage"
            
        action_distribution = {k: v / total for k, v in action_distribution.items()}
        sampled_action = np.random.choice(
            list(action_distribution.keys()),
            size=1,
            p=list(action_distribution.values()),
        )[0]
        return sampled_action

    def dijkstra(self, graph, start_node, target_node):
        distances = {node: float("infinity") for node in graph}
        distances[start_node] = 0
        pq = [(0, start_node)]
        visited = set()
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_node == target_node:
                return current_distance
            if current_node in visited:
                continue
            visited.add(current_node)
            if current_node in graph:
                for neighbor, weight in graph[current_node].items():
                    if neighbor not in visited:
                        distance = current_distance + weight
                        if distance < distances.get(neighbor, float('infinity')):
                            distances[neighbor] = distance
                            heapq.heappush(pq, (distance, neighbor))
        return float("infinity")

    def update_state(self, context, state_dict):
        c_state = state_dict.get("consistent_state", self.initial_stage)
        engagement = state_dict.get("consistent_engagement", self.receptivity)
        error_count = state_dict.get("consistent_error_count", 0)

        if c_state == "Contemplation":
            if len(self.beliefs) == 0:
                c_state = "Preparation"
            state_dict["consistent_state"] = c_state
            return None
        elif c_state == "Preparation":
            return None
        else:
            top_topics = self.top5_related_topics(context)
            if not top_topics:
                return None
            predicted_topic = top_topics[0]
            if len(self.engagemented_topics) > 0 and predicted_topic == self.engagemented_topics[0]:
                engagement = 4
                error_count = 0
                motivation_analysis = self.verify_motivation(context)
                if "yes" in motivation_analysis.lower():
                    c_state = "Motivation"
            else:
                dist = float('inf')
                if len(self.engagemented_topics) > 0:
                    dist = self.dijkstra(self.topic_graph, self.engagemented_topics[0], predicted_topic)
                if dist <= 3:
                    engagement = 3
                    error_count = 0
                elif dist <= 5:
                    engagement = 2
                else:
                    engagement = 1
                    if len(context) > 10:
                        error_count += 1
            
            state_dict["consistent_state"] = c_state
            state_dict["consistent_engagement"] = engagement
            state_dict["consistent_error_count"] = error_count
            return f"The client's perceived topic is {predicted_topic}."

    def select_information(self, action, context):
        messages = []
        if "?" not in context[-1]:
            return None
            
        prompt = """Here is a conversation between Client and Counselor:
[@conv]

Is there a question in the last utterance of Counselor? Yes or No"""
        prompt = prompt.replace("[@conv]", "\n".join(context[-3:]))
        response = "Yes, there is a question in the last utterance of Counselor."
        messages.append(HumanMessage(content=prompt))
        messages.append(AIMessage(content=response))
        
        if action == "Inform":
            prompt2 = """Can the following Client's persona answer the question? Yes or No
[@persona]"""
            personas = self.personas
        elif action == "Downplay":
            prompt2 = """Can the following Client's persona reply the question to downplay the importance or impact of behavior? Yes or No
[@persona]"""
            personas = self.beliefs
        elif action == "Blame":
            prompt2 = """Can the following Client's persona reply the question to blame external factors or others to justify? Yes or No
[@persona]"""
            personas = self.beliefs
        elif action == "Hesitate":
            prompt2 = """Can the following Client's persona reply the question to show uncertainty, indicating ambivalence about change? Yes or No
[@persona]"""
            personas = self.beliefs
        else:
            return None
            
        for persona in personas:
            prompt_i = prompt2.replace("[@persona]", persona)
            msgs_i = messages + [HumanMessage(content=prompt_i)]
            response = self._call_llm(msgs_i, temperature=0.2)
            if response and "yes" in response.lower():
                if action == "Hesitate":
                    personas.pop(personas.index(persona))
                return persona
                
        if len(personas) > 0:
            persona = random.choice(personas)
            if action == "Hesitate":
                personas.pop(personas.index(persona))
            return persona
            
        return None

    def get_engage_instruction(self, state_dict):
        engagement = state_dict.get("consistent_engagement", self.receptivity)
        topics = self.engagemented_topics
        if not topics:
            return None
        if engagement == 1:
            return "You should provide vague and broad answers that avoid focusing on the current topic. Shift the conversation subtly toward unrelated areas, without engaging deeply with the topic."
        elif engagement == 2 and len(topics) > 2:
            return f"Acknowledge the importance of {topics[2]}, but hint that your focus is on a more specific topic, i.e. {topics[1]} within it."
        elif engagement == 3 and len(topics) > 1:
            return f"Engage more directly with {topics[1]}, and offer responses that subtly indicate there’s a deeper, more specific issue worth exploring within that topic, i.e. {topics[0]}."
        elif engagement == 4 and len(topics) > 0:
            return f"Offer specific responses that affirm the counselor is on the right track, showing that you're motivated by {topics[0]}. {self.motivation}"
        return None


@traced_span("consistent_client")
def consistent_client_node(state: Dict[str, Any], config: RunnableConfig, runtime: Runtime) -> Dict[str, Any]:
    logger.info("consistent_client_node invoked")
    
    configurable = config.get("configurable", {})
    patient_profile_data_dict = configurable.get("patient_profile_data")
    profile_data = PatientProfileData.model_validate(patient_profile_data_dict)
    
    # Read memory from inter-session
    store = runtime.store
    memory_item = store.get(PATIENT_MEMORY_NS, "consistent_memory")
    memory_prompt = memory_item.value.get("summary", "") if memory_item else ""
    
    # Ensure state tracking fields exist in state
    if "consistent_client_state" not in state:
        state["consistent_client_state"] = {}
        
    c_state_dict = state["consistent_client_state"]
    
    adapter = ConsistentClientAdapter(profile_data, c_state_dict, memory_prompt)
    
    all_messages = state.get("messages", [])
    patient_turns = sum(1 for m in all_messages if m.name == "Patient") + 1
    
    # Build context array for SOTA
    context = []
    for msg in all_messages[-10:]:
        if msg.name == "Therapist":
            context.append(f"Counselor: {msg.content}")
        elif msg.name == "Patient":
            context.append(f"Client: {msg.content}")

    if not context:
        context.append("Counselor: Hello. How are you?")
    
    engagement_analysis = adapter.update_state(context, c_state_dict)
    
    current_sota_state = c_state_dict.get("consistent_state", adapter.initial_stage)
    logger.info(f"Consistent Client State: {current_sota_state} | Engagement: {c_state_dict.get('consistent_engagement', adapter.receptivity)}")
    if engagement_analysis:
        logger.info(f"Consistent Client Topic Analysis: {engagement_analysis}")
    
    information = None
    if current_sota_state == "Motivation":
        engage_instruction = f"Offer specific responses that affirm the counselor is on the right track, showing that you're motivated by {adapter.engagemented_topics[0]}."
        instruction = f"[{adapter.motivation} {adapter.action2prompt['Acknowledge']} {engage_instruction}]"
        c_state_dict["consistent_state"] = "Contemplation"
        action = "Acknowledge"
    elif current_sota_state == "Precontemplation":
        engage_instruction = adapter.get_engage_instruction(c_state_dict)
        error_count = c_state_dict.get("consistent_error_count", 0)
        if error_count >= 5:
            action = "Terminate"
        else:
            action = adapter.select_action("\n".join(context[-3:]), c_state_dict)
            
        if action in ["Inform", "Downplay", "Blame"]:
            information = adapter.select_information(action, context)
            instruction = f"[{engage_instruction} {adapter.state2prompt[current_sota_state]} {adapter.action2prompt.get(action, '')} You should follow the persona: {information} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
        else:
            instruction = f"[{engage_instruction} {adapter.state2prompt[current_sota_state]} {adapter.action2prompt.get(action, '')} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
    elif current_sota_state == "Contemplation":
        action = adapter.select_action("\n".join(context[-3:]), c_state_dict)
        if action in ["Hesitate", "Inform"]:
            information = adapter.select_information(action, context)
            instruction = f"[{adapter.state2prompt[current_sota_state]} {adapter.action2prompt.get(action, '')} You should follow the persona: {information} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
        else:
            instruction = f"[{adapter.state2prompt[current_sota_state]} {adapter.action2prompt.get(action, '')} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
    else:
        if len(adapter.acceptable_plans) == 0:
            action = "Terminate"
        else:
            action = adapter.select_action("\n".join(context[-3:]), c_state_dict)
        
        if action == "Inform":
            information = adapter.select_information(action, context)
            instruction = f"[{adapter.state2prompt[current_sota_state]} {adapter.action2prompt.get(action, '')} You should follow the persona: {information} Don't show overknowledge and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
        elif action == "Plan":
            information = adapter.acceptable_plans.pop(0) if adapter.acceptable_plans else ""
            instruction = f"[{adapter.state2prompt[current_sota_state]} {information} {adapter.action2prompt.get(action, '')} Don't show overknowledge, and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"
        else:
            instruction = f"[{adapter.state2prompt[current_sota_state]} {adapter.action2prompt.get(action, '')} Don't show overknowledge, and keep your responses concise (no more than 50 words). Don't highlight your state explicitly.]"

    instruction = instruction.replace("\n", " ")
    logger.info(f"Consistent Client Action chosen: {action}")
    logger.info(f"Consistent Client LLM Instruction: {instruction}")
    
    memory_section = f"\nMemory from previous sessions:\n{memory_prompt}\n" if memory_prompt else ""
    
    system_prompt = f"""In this role-play scenario, you'll take on the role of a Client discussing about your {adapter.behavior} where the Counselor's goal is {adapter.goal}.

Here is your personas which you need to follow consistently throughout the conversation:
{chr(10).join(['- ' + p for p in adapter.personas])}
{chr(10).join(['- ' + b for b in adapter.beliefs])}
{memory_section}
Please follow these guidelines in your responses:
- **Start your response with "Client: "**
- **Adhere strictly to the state, action and persona specified within square brackets.**
- **Keep your responses coherent and concise, and no more than 3 sentences.**
- **Be natural and concise without being overly polite.**
- **Stick to the persona provided and avoid introducing contradictive details.**
"""
    
    # Build alternating message history matching the original self.messages structure:
    # [system] + mock initial exchange + [user/assistant]* for history + [user: last_counselor + instruction]
    messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content="Counselor: Hello. How are you?"))
    messages.append(AIMessage(content="Client: I am good. What about you?"))
    for turn in context[:-1]:
        if turn.startswith("Counselor:"):
            messages.append(HumanMessage(content=turn))
        else:
            messages.append(AIMessage(content=turn))
    messages.append(HumanMessage(content=f"{context[-1]} {instruction}"))

    response = adapter._call_llm(messages)

    response = response.replace("\n", " ").strip().lstrip()
    if not response.startswith("Client: "):
        response = f"Client: {response}"
    if "Counselor: " in response:
        response = response.split("Counselor: ")[0].strip()
    response = response.removeprefix("Client:").strip()
    
    patient_message = AIMessage(content=response, name="Patient")
    summary_line = format_speaker_utterance(response, PATIENT_PREFIX)
    logger.info(f"\n\nConsistent Client response:\n\n{summary_line}\n\n")
    
    current_turns = state.get("current_session_turns", [])
    turn_record = {
        "turn_number": len(current_turns) + 1,
        "speaker": "patient",
        "volley": response,
        "current_dominant_stage": current_sota_state,
    }
    
    c_state_dict["remaining_personas"] = adapter.personas
    c_state_dict["remaining_beliefs"] = adapter.beliefs
    c_state_dict["remaining_plans"] = adapter.acceptable_plans
    
    return {
        "messages": state.get("messages", []) + [patient_message],
        "current_speaker": "patient",
        "turn_count": state.get("turn_count", 0) + 1,
        "current_session_turns": current_turns + [turn_record],
        "consistent_client_state": c_state_dict,
    }
