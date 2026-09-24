"""
Store namespace constants for LangGraph cross-session memory.

All nodes that access the InMemoryStore use these namespace tuples so that
keys are consistently namespaced per patient.
"""
import os

_PATIENT_ID: str = os.getenv("PATIENT_ID", "117")

# ── Patient memory ──────────────────────────────────────────────
# Private to the patient. Therapist agent MUST NOT read from this namespace.
# Keys:
#   "evolving_decisional_balance"
#   "facts"
#   "agreed_change_plans"
#   "implicit_threads"
#   "session_episodic_summary"
#   "past_event_titles"
#   "chronological_diary"
#   "motivations"
#   "concerns"
#   "persona_and_stylistic_baseline"
PATIENT_MEMORY_NS: tuple[str, str] = (_PATIENT_ID, "patient_memory")


# ── Therapist working memory ──────────────────────────────────────────────
# Therapist's private rolling memory.
# Keys:
#   "biographical_facts"
#   "core_values"
#   "target_behavior"
#   "darn_change_talk"
#   "cat_change_talk"
#   "sustain_talk_themes"
#   "change_plan"
#   "current_clinical_focus"
THERAPIST_MEMORY_NS: tuple[str, str] = (_PATIENT_ID, "therapist_memory")


def dump_store(store, filepath: str) -> None:
    """Serialize an InMemoryStore to a JSON file."""
    import json
    import os
    data = []
    # BaseStore.search returns all items when given empty tuple
    for item in store.search(()):
        data.append({
            "namespace": list(item.namespace),
            "key": item.key,
            "value": item.value
        })
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_store(store, filepath: str) -> None:
    """Populate an InMemoryStore from a JSON file."""
    import json
    import os
    if not os.path.exists(filepath):
        return
    with open(filepath, "r") as f:
        data = json.load(f)
    for item in data:
        store.put(tuple(item["namespace"]), item["key"], item["value"])
