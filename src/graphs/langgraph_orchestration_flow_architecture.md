# LangGraph Orchestration Flow (High-Level Architecture)

This document explains the conceptual flow and architecture of the therapy simulation. Rather than focusing on code-level types and state dictionaries, it maps out how the simulation orchestrates the passage of time, the interactions between agents, and the evolution of the patient's psyche.

## 1. The Core Concept: A Hierarchical Architecture
The simulation is built on a **two-tier architecture**:
*   **Micro-Level (Single Session Graph):** Manages the immediate, turn-by-turn conversation between the Therapist and the Patient. It focuses on the real-time dynamics of a single appointment.
*   **Macro-Level (Multi-Session Graph):** Manages the passage of time across an entire therapy program (typically 4 sessions). It handles long-term memory consolidation and simulates what happens to the patient *between* sessions.

---

## 2. Micro-Level Flow: Inside a Single Session
When a single session is running, the orchestration follows a continuous loop. Because LLMs lack a natural sense of time and tend to converse indefinitely, the orchestrator actively manages the clock and forces phase transitions.

**The Turn-by-Turn Cycle:**
1.  **Dynamic Instructions (The Clock):** Before the Therapist speaks, the orchestrator checks the turn count. As the session nears its maximum length, it injects mandatory "System Alerts" into the Therapist's prompt. These dynamic instructions force the Therapist to issue verbal time warnings (e.g., "We have a few minutes left") and ultimately execute a graceful, hard stop without asking further questions.
2.  **Therapist Action:** The Therapist agent analyzes the conversation, adheres to any dynamic time-warnings, and generates a clinical response.
    *   *Adversarial Override (MIIN Injection):* On specific, predetermined turns, the orchestration intercepts the normal therapist and forces an intentionally "bad" or resistive response (e.g., a "Righting Reflex") to stress-test the patient's psychological realism.
3.  **Cognitive Evaluation (The "Judge"):** Before the patient even "hears" the therapist's response, an invisible evaluator (`cognitive_interpreter`) analyzes the therapist's intervention. It scores whether the intervention was clinically sound and decides its immediate semantic impact on the patient.
4.  **Patient Reaction:** The Patient agent receives the therapist's words along with the judge's evaluation, uses its internal "Thinking Framework" (Cognitive and Psychological layers), and generates its spoken reply.
5.  **Rolling Memory Trigger:** Because the LLM cannot hold an infinite transcript in its active context, the graph periodically pauses the main dialogue (e.g., every 6 turns) to trigger the **Rolling Memory Agents**. These agents compress the recent dialogue into long-term summary insights, keeping the prompt context window efficient.
6.  **Session End Check:** If the hard turn limit is reached and the dynamic wrap-up is complete, the orchestrator breaks the loop and passes control back to the Macro-Level.

---

## 3. Macro-Level Flow: Across Multiple Sessions
Once a session concludes, the parent graph takes over to handle the transition to the next appointment.

**The Transition Cycle:**
1.  **Final Memory Consolidation:** The remaining un-summarized turns of the completed session are passed to the Rolling Memory agents for a final "end-of-session" wrap-up for both the Patient and the Therapist.
2.  **Macro-Shift Application:** The overarching psychological impact of the entire session is calculated. This results in permanent numeric shifts to the patient's core latent variables (e.g., their baseline Anger or Self-Efficacy changes based on the session's success).
3.  **Between-Session Simulation (The Sleeper Effect):** Therapy doesn't only happen in the clinic. Before the next session begins, the `life_event_simulator` runs. It generates a realistic life event that happens to the patient during the week (e.g., "Had a fight with a spouse"). This event adjusts the patient's latent variables, meaning they might return to the next session in a different state of mind than when they left.
4.  **Session Increment:** The system archives the complete transcript, clears the short-term conversation context, and starts the Single-Session loop again for Session N+1, carrying over only the long-term memory and the newly updated latent variables.