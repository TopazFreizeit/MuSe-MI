# Therapist Agent Architecture

This document maps out the Input/Output architecture of the three specialized LLM sub-agents that power the Therapist. Every variable listed here serves a specific, necessary role in maintaining clinical fidelity, preventing hallucinations, and ensuring adherence to Motivational Enhancement Therapy (MET) protocols.

---

## 1. Clinical Analyzer
**Purpose:** Diagnoses the patient's motivational state based on their most recent turn and recent history.

### Inputs
- **`<conversation_history>`**: The rolling window of previous conversation turns (excluding the final exchange). *Necessary to detect temporal patterns like repetitive loops or sustained resistance.*
- **`<last_exchange>`**: The previous therapist turn and the current patient turn. *Necessary to pinpoint exactly what intervention the patient is reacting to.*

### Outputs (JSON `patient_diagnosis`)
- **`classification_reasoning`**: Step-by-step reasoning for language classification. *Necessary (Chain of Thought) to prevent the LLM from mislabeling complex patient statements.*
- **`resistance_detected`**: Verbatim sustain talk or 'none'. *Necessary to inform the Strategist when the patient is pushing back.*
- **`preparatory_change_talk`**: Verbatim DARN talk or 'none'. *Necessary to track early motivational shifts without falsely assuming full commitment.*
- **`commitment_and_action_talk`**: Verbatim CAT or 'none'. *Necessary to identify when the patient is ready to make concrete behavioral changes.*
- **`stage_of_change`**: Categorical TTM stage (Precontemplation, Contemplation, Action). *Necessary only for downstream research analytics. It is excluded from the Strategist's prompt to prevent logical contradictions with the raw behavioral flags.*
- **`planning_reasoning_trace`**: Reasoning for planning readiness. *Necessary (CoT) to ensure the patient has actually shown commitment before transitioning.*
- **`ready_for_planning`**: Boolean flag. *Necessary to trigger the Strategist's transition into the "Planning" phase.*
- **`stuckness_reasoning_trace`**: Reasoning for resistance loops. *Necessary (CoT) to accurately detect if the conversation is spinning in circles.*
- **`patient_is_stuck`**: Boolean flag. *Necessary to alert the Strategist to change tactics (e.g., using Complex Reflection or Shifting Focus) instead of arguing.*
- **`guidance_reasoning_trace`**: Reasoning for guidance requests. *Necessary (CoT) to verify if the patient genuinely asked a question.*
- **`patient_requests_guidance_or_info`**: Boolean flag. *Necessary to grant the Strategist permission to use the Give Information (GI) or Advise With Permission (ADP) techniques, which are otherwise restricted.*

---

## 2. Clinical Strategist
**Purpose:** Determines the overarching clinical intent and selects the exact MET techniques allowed for the current turn.

### Inputs
- **Patient Context Blocks**: `biographical_facts`, `core_values`, `target_behavior`, `change_plan`, `current_clinical_focus`. *Necessary to ground the strategy in the patient's specific life and goals.*
- **`patient_diagnosis`**: The JSON output from the Analyzer, excluding the `stage_of_change`. *Necessary to tailor the strategy purely to the patient's objective, real-time behavioral language rather than a subjective theoretical label.*
- **`mobilizing_ct_count_this_session`**: Counter of CAT instances. *Necessary to gauge cumulative readiness over the course of the current session.*
- **`<conversation_history>` & `<last_exchange>`**: The conversational text. *Necessary to provide semantic context for the strategy.*
- **`### System Alert`**: Dynamic instructions injected at the very beginning or end of a session. *Necessary to enforce strict structural boundaries (e.g., wrapping up the session gracefully).*

### Outputs (JSON)
- **`strategist_reasoning_trace`**: Step-by-step thinking connecting the diagnosis to a strategy. *Necessary (CoT) to ensure the chosen techniques logically match the patient's state.*
- **`current_mi_phase`**: Engaging, Focusing, Evoking, or Planning. *Necessary to restrict the available technique pool to those appropriate for the current phase.*
- **`macro_intent`**: A plain-text description of the overarching goal (e.g., "Explore the patient's reluctance"). *Necessary to give the Formulator thematic direction beyond just a mechanical technique code.*
- **`allowed_techniques`**: A constrained list of 1-3 valid MET technique codes (e.g., `["OQ", "CR"]`). *Necessary to strictly prevent the Formulator from generating non-therapeutic or aggressive responses.*

---

## 3. Formulator
**Purpose:** Drafts the actual conversational response using the constraints provided by the Strategist.

### Inputs
- **`macro_intent` & `allowed_techniques`**: The strict constraints from the Strategist. *Necessary to force the LLM to write a response that adheres to MET principles.*
- **`Clinical Context`**: Dynamic contextual reasoning traces (e.g., Stuckness, Planning readiness). *Necessary to explain to the Formulator exactly **why** a certain technique was chosen (e.g., what the patient is stuck on), allowing it to craft a precise, empathetic response.*
- **Patient Context Blocks**: Target behavior, change plans, etc. *Necessary so the LLM can reference specific facts (like a spouse's name or a specific plan) in its natural language response.*
- **`<conversation_history>` & `<last_exchange>`**: The conversational text. *Necessary so the generated response flows naturally from the immediate prior turn.*
- **`### System Alert`**: Structural warnings. *Necessary so the Formulator knows, for example, not to ask an open question on the very last turn of the session.*

### Outputs (JSON)
- **`reasoning_chain`**: Step-by-step explanation starting with a mandatory *context evaluation* (to prevent defaulting to the easiest technique), followed by technique selection, drafting, and structural verification. *Necessary (CoT) to guarantee the Formulator deeply considers the clinical context before drafting its response.*
- **`chosen_technique`**: The single MET code the Formulator decided to execute from the allowed list. *Necessary for analytics, logging, and graphing therapist adherence.*
- **`final_response`**: The verbatim string that will be spoken to the patient. *Necessary to continue the simulation loop.*
