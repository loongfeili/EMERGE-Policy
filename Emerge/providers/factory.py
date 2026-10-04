"""Create configured LLM provider instances for agent runtimes."""

from __future__ import annotations

from Emerge.config.schema import Config
from Emerge.providers.base import GenerationSettings, LLMProvider


def create_provider(
    config: Config,
    *,
    model: str | None = None,
) -> LLMProvider:
    """Create the provider selected by ``model`` and the runtime config."""
    selected_model = model or config.agents.defaults.model
    provider_name = config.get_provider_name(selected_model)
    provider_config = config.get_provider(selected_model)

    if provider_name == "openai_codex" or selected_model.startswith(
        "openai-codex/"
    ):
        from Emerge.providers.openai_codex_provider import (
            OpenAICodexProvider,
        )

        provider: LLMProvider = OpenAICodexProvider(
            default_model=selected_model
        )
    elif provider_name == "responses":
        from Emerge.providers.responses_provider import ResponsesProvider

        if not provider_config or not provider_config.api_key or not provider_config.api_base:
            raise ValueError("Responses requires api_key and api_base in providers.responses")
        provider = ResponsesProvider(
            api_key=provider_config.api_key,
            api_base=provider_config.api_base,
            default_model=selected_model,
            api_base_fallbacks=provider_config.api_base_fallbacks,
            extra_headers=provider_config.extra_headers,
            reasoning_summary=provider_config.reasoning_summary,
            trust_env=provider_config.trust_env,
        )
    elif provider_name == "custom":
        from Emerge.providers.custom_provider import CustomProvider

        if (config.get_api_base(selected_model) or "").rstrip("/").endswith("/responses"):
            from Emerge.providers.responses_provider import ResponsesProvider
            provider_class = ResponsesProvider
            endpoint_options = {
                "api_base_fallbacks": provider_config.api_base_fallbacks,
                "extra_headers": provider_config.extra_headers,
                "reasoning_summary": provider_config.reasoning_summary,
                "trust_env": provider_config.trust_env,
            }
        else:
            provider_class = CustomProvider
            endpoint_options = {}
        provider = provider_class(
            api_key=(provider_config.api_key or "no-key"),
            api_base=(
                config.get_api_base(selected_model)
                or "http://localhost:8000/v1"
            ),
            default_model=selected_model,
            **endpoint_options,
        )
    elif provider_name == "azure_openai":
        from Emerge.providers.azure_openai_provider import (
            AzureOpenAIProvider,
        )

        if (
            provider_config is None
            or not provider_config.api_key
            or not provider_config.api_base
        ):
            raise ValueError(
                "Azure OpenAI requires api_key and api_base in "
                "providers.azure_openai"
            )
        provider = AzureOpenAIProvider(
            api_key=provider_config.api_key,
            api_base=provider_config.api_base,
            default_model=selected_model,
            api_version=provider_config.api_version,
            max_tokens_parameter=provider_config.max_tokens_parameter,
            extra_headers=provider_config.extra_headers,
            trust_env=provider_config.trust_env,
        )
    else:
        from Emerge.providers.litellm_provider import LiteLLMProvider
        from Emerge.providers.registry import find_by_name

        provider_spec = find_by_name(provider_name)
        has_credentials = provider_config and provider_config.api_key
        needs_api_key = not (
            selected_model.startswith("bedrock/")
            or has_credentials
            or (
                provider_spec
                and (provider_spec.is_oauth or provider_spec.is_local)
            )
        )
        if needs_api_key:
            raise ValueError(
                f"No API key configured for model {selected_model!r}"
            )
        provider = LiteLLMProvider(
            api_key=(
                provider_config.api_key if provider_config else None
            ),
            api_base=config.get_api_base(selected_model),
            default_model=selected_model,
            extra_headers=(
                provider_config.extra_headers if provider_config else None
            ),
            provider_name=provider_name,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider
