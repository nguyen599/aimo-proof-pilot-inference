"""vLLM registration for Proof Pilot OLMo3Sink checkpoints."""

from __future__ import annotations

_REGISTERED = False


def register() -> None:
    """Register the OLMo3Sink target model in every vLLM process."""
    global _REGISTERED
    if _REGISTERED:
        return

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Olmo3SinkForCausalLM",
        "olmo3_sink_vllm.model:Olmo3SinkForCausalLM",
    )
    _REGISTERED = True


__all__ = ["register"]
