# Patient Agent Architecture

This document maps out the Input/Output architecture of the Simulated Patient. The architecture uses a "Thinking Framework" (Cognitive and Psychological layers) to prevent positivity bias and ensure the agent's spoken responses are strictly grounded in its current latent clinical state and core persona.

---

## 1. Patient Profile Initializer
**Purpose:** Loads and normalizes the static clinical baseline and demographic persona of the patient before the simulation begins.

### Inputs
- **Raw JSON Profile**: Data extracted from empirical datasets (e.g., `profiles.jsonl`), including raw scores for clinical variables.
- **`scale_boundaries.md`**: The authoritative statistical bounds used to translate raw questionnaire scores into normalized 1-100 scales.

### Outputs
- **`PatientProfileData`**: A structured object containing the patient's Topic, Behavior, Personas, Beliefs, Motivation, and Acceptable Plans. *Necessary to provide the LLM with a consistent, static identity across all sessions.*
- **`PatientLatentVariables`**: The initial 1-100 scores for `anger`, `problem_recognition`, `motivational_readiness`, and `self_efficacy`. *Necessary as the starting point for the Cognitive Simulator.*

---

## 2. Patient Planner & Prompt Builder
**Purpose:** Constructs the precise context window for the LLM on every turn, translating numerical data into qualitative clinical instructions.

### Inputs
- **`PatientLatentVariables` (Numeric)**: The current 1-100 scores.
- **Clinical Rubrics**: Functions that translate scores (e.g., <=33.3, <=66.6, >66.6) into semantic labels (e.g., "High (Defensive and Hostile)"). *Necessary because LLMs struggle to roleplay abstract numbers; they need concrete, behavioral text instructions based on those numbers.*
- **`dynamic_directives`**: Recent macro-shifts calculated by the Patient State Manager. *Necessary to inform the patient how their mindset has shifted due to recent therapist interventions.*
- **`<conversation_history>` & `<last_exchange>`**: The dialogue context.

### Outputs
- **System Prompt**: A complete, compiled string injected into the LLM, containing strict instructions on tone, current emotional state, and the "Thinking Framework" requirements.

---

## 3. Cognitive Response Generator (The LLM)
**Purpose:** Generates the patient's internal reasoning and final spoken response based on the prompt constraints.

### Outputs (JSON `PatientCognitiveDirective`)
- **`cognitive_layer.decision_rule`**: The heuristic the patient uses to decide whether to engage or deflect. *Necessary (CoT) to force the LLM to plan its reaction logically before generating dialogue.*
- **`cognitive_layer.optimizing_for`**: What the patient is trying to protect or achieve. *Necessary to maintain psychological realism and prevent the LLM from simply being "helpful."*
- **`conversational_stance`**: Explicit stance based on frustration and ambivalence. *Necessary to dictate the immediate tone.*
- **`psychological_layer.defense_pattern`**: The specific MI resistance behavior active right now. *Necessary to categorize the resistance for downstream analytics.*
- **`psychological_layer.core_paradox`**: The central ambivalence driving the patient (Change Talk vs Sustain Talk). *Necessary to explicitly map the tension of ambivalence.*
- **`active_reasoning`**: Step-by-step reasoning in-character. *Necessary (CoT) to bridge the cognitive rules with the actual words spoken.*
- **`pressing_disclosure_intention`**: What the patient urgently wants to discuss or hide.
- **`response`**: The exact 1-2 sentence spoken reply. *Necessary to continue the simulation loop.*