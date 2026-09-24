"""Therapist agent package."""

__all__ = ['therapist_node']


def __getattr__(name):
    if name == "therapist_node":
        from .therapist_agent import therapist_node
        return therapist_node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
