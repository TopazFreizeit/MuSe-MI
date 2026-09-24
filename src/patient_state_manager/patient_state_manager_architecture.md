# Patient State Manager Architecture

This document maps out the architecture of the Patient State Manager. Unlike the turn-by-turn dialogue generation, this module is responsible for the longitudinal, multi-turn, and multi-session evolution of the patient. It bridges the gap between semantic dialogue and numerical latent variables.

---

## 1. Patient State Updater (The LLM Evaluator)
**Purpose:** Periodically analyzes chunks of the transcript (e.g., every X turns) to determine if a significant psychological shift has occurred based on the therapist's interventions.

### Inputs
- **`<transcript_chunk>`**: A recent segment of the conversation. *Necessary to evaluate the flow and impact of interventions over a small window rather than a single turn.*
- **Current `PatientLatentVariables`**: The numerical state before the chunk occurred. *Necessary to provide a baseline for comparison.*
- **Therapist Interventions**: A list of the specific MI techniques used by the therapist during this chunk. *Necessary to evaluate cause-and-effect (e.g., did a Complex Reflection lead to an insight?).*

### Outputs (JSON `PatientStateUpdateDTO`)
- **`reasoning_general`**: A broad evaluation of the therapist's latest interaction against the patient's decisional balance. *Necessary (CoT) to prevent hallucinated state changes.*
- **`anger`**: An object containing `reasoning` and `semantic_impact` (Success/Neutral/Resistance). *Necessary to translate qualitative textual analysis into a structured Enum for the Math Engine.*
- **`motivational_readiness`**: An object containing `reasoning` and `semantic_impact` (Success/Neutral/Resistance).
- **`self_efficacy`**: An object containing `reasoning` and `semantic_impact` (Success/Neutral/Resistance).
- **`problem_recognition`**: An object containing `reasoning` and `semantic_impact` (Success/Neutral/Resistance).

---

## 2. Deterministic Cognitive Simulator (The Math Engine)
**Purpose:** Translates the categorical Enums from the LLM Evaluators into precise numerical deltas and applies them to the patient's latent variables.

### Inputs
- **Current `PatientLatentVariables`**: The raw 1-100 numerical scores.
- **`SemanticImpact` Enums**: Short-term impacts (Success/Neutral/Resistance) from the State Updater.
- **`MacroShiftSemanticImpact` Enums**: Long-term impacts (Major/Minor Progress/Regression) from the Rolling Memory Agent.

### Outputs
- **Updated `PatientLatentVariables`**: The newly calculated 1-100 numerical scores. *Necessary to guarantee that mathematical bounds (1-100) are strictly respected without relying on the LLM to do math. This deterministic layer prevents the LLM from outputting impossible scores or wild fluctuations.*

---

## 3. Rolling Memory Agent
**Purpose:** Synthesizes and stores long-term clinical narrative context to pass between large context windows or between distinct sessions.

### Inputs
- **Full Session Transcript**: The raw dialogue history.
- **`PatientLatentVariables`**: The numerical state at the time of summarization.
- **Prior Memory Items**: Previously stored facts, plans, threads, motivations, and concerns.

### Outputs (JSON `RollingMemoryOutput`)
- **`memory_update_reasoning`**: Explicit rationale for every change made to the memory record. *Necessary for evaluating the LLM's memory component performance.*
- **`facts`**: A list of specific, immutable declarative anchors. (Accumulative)
- **`agreed_change_plans`**: A list of `ChangePlanStatusItem` objects detailing intentions and attempt status. (Accumulative)
- **`implicit_threads`**: Deferred topics the patient hinted at. (Accumulative)
- **`motivations`**: Top 3-5 active Change Talk drivers. (Replaced)
- **`concerns`**: Top 3-5 active Sustain Talk drivers. (Replaced)
- **`persona_and_stylistic_baseline`**: Description of the patient's communication style. (Replaced)
- **`macro_shift`**: A `MacroShift` object detailing the `reasoning` and `semantic_impact` (Major/Minor Progress/Regression) for `anger`, `self_efficacy`, `problem_recognition`, and `motivational_readiness`. *Necessary to calculate long-term state shifts across an entire session or window.*