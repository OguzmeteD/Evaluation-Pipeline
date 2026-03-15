from __future__ import annotations

import hashlib
from typing import Any

from src.core.langfuse_client import LangfuseCollectorClient
from src.schemas.experiment_runner import (
    PromptPublishTarget,
    PublishedPromptRequest,
    PublishedPromptResult,
    PromptResolutionRequest,
    PromptResolutionResult,
    PromptSource,
    PromptType,
    ResolvedPrompt,
)


class PromptResolverService:
    def __init__(self, collector: LangfuseCollectorClient) -> None:
        self.collector = collector

    def resolve_prompt(self, request: PromptResolutionRequest) -> PromptResolutionResult:
        if request.source == PromptSource.CUSTOM_PROMPT:
            custom_prompt = (request.custom_prompt or "").strip()
            return PromptResolutionResult(
                resolved_prompt=ResolvedPrompt(
                    source=PromptSource.CUSTOM_PROMPT,
                    target=request.target,
                    prompt_type=request.prompt_type,
                    compiled_text=custom_prompt,
                    messages=[],
                    variables=[],
                    is_fallback=False,
                    fingerprint=_fingerprint(custom_prompt),
                ),
                found=True,
                warnings=[],
            )

        warnings: list[str] = []
        try:
            prompt_client = self.collector.get_prompt(
                name=request.prompt_name or "",
                version=request.prompt_version,
                label=request.prompt_label,
                type=request.prompt_type.value,
            )
            resolved = self._normalize_langfuse_prompt(prompt_client=prompt_client, request=request)
            return PromptResolutionResult(resolved_prompt=resolved, found=True, warnings=warnings)
        except Exception as exc:
            custom_prompt = (request.custom_prompt or "").strip()
            if not custom_prompt:
                raise ValueError(f"Prompt bulunamadi ve custom fallback verilmedi: {exc}") from exc
            warnings.append(f"Langfuse prompt cekilemedi, custom prompt fallback kullanildi: {exc}")
            return PromptResolutionResult(
                resolved_prompt=ResolvedPrompt(
                    source=PromptSource.CUSTOM_PROMPT,
                    target=request.target,
                    prompt_name=request.prompt_name,
                    prompt_label=request.prompt_label,
                    prompt_version=request.prompt_version,
                    prompt_type=request.prompt_type,
                    compiled_text=custom_prompt,
                    messages=[],
                    variables=[],
                    is_fallback=True,
                    fingerprint=_fingerprint(custom_prompt),
                ),
                found=False,
                warnings=warnings,
            )

    def publish_prompt(self, request: PublishedPromptRequest) -> PublishedPromptResult:
        if request.prompt_type == PromptType.CHAT and request.messages:
            prompt_payload: str | list[dict[str, Any]] = request.messages
        else:
            prompt_payload = (request.prompt_text or "").strip()
        if not prompt_payload:
            raise ValueError("Publish icin prompt icerigi zorunlu.")
        prompt_client = self.collector.create_or_update_prompt_version(
            name=request.prompt_name,
            prompt=prompt_payload,
            prompt_type=request.prompt_type.value,
            label=request.label,
            commit_message=request.commit_message,
            tags=[request.target.value, "codex-studio", "published"],
        )
        compiled_text = ""
        messages: list[dict[str, Any]] = []
        if request.prompt_type == PromptType.CHAT:
            compiled_messages = list(getattr(prompt_client, "prompt", request.messages))
            try:
                compiled_messages = list(prompt_client.compile())
            except Exception:
                pass
            messages = [self._normalize_chat_message(message) for message in compiled_messages]
            compiled_text = self._stringify_chat_messages(messages)
        else:
            compiled_text = getattr(prompt_client, "prompt", prompt_payload)
            try:
                compiled_text = prompt_client.compile()
            except Exception:
                pass
        return PublishedPromptResult(
            target=request.target,
            prompt_name=getattr(prompt_client, "name", request.prompt_name),
            prompt_version=getattr(prompt_client, "version", None),
            prompt_label=request.label,
            prompt_type=request.prompt_type,
            source=request.source,
            fingerprint=_fingerprint(compiled_text),
            use_for_next_run=request.use_for_next_run,
        )

    def list_prompts(
        self,
        *,
        name: str | None = None,
        label: str | None = None,
        tag: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.collector.list_prompts(name=name, label=label, tag=tag, limit=limit)

    def _normalize_langfuse_prompt(
        self,
        *,
        prompt_client: Any,
        request: PromptResolutionRequest,
    ) -> ResolvedPrompt:
        compiled_text = ""
        messages: list[dict[str, Any]] = []
        variables = list(getattr(prompt_client, "variables", []))
        if request.prompt_type == PromptType.TEXT:
            raw_prompt = getattr(prompt_client, "prompt", "")
            try:
                compiled_text = prompt_client.compile()
            except Exception:
                compiled_text = raw_prompt
            messages = []
        else:
            raw_messages = list(getattr(prompt_client, "prompt", []))
            compiled_messages = raw_messages
            try:
                compiled_messages = list(prompt_client.compile())
            except Exception:
                compiled_messages = raw_messages
            messages = [self._normalize_chat_message(message) for message in compiled_messages]
            compiled_text = self._stringify_chat_messages(messages)

        return ResolvedPrompt(
            source=PromptSource.LANGFUSE_PROMPT,
            target=request.target,
            prompt_name=getattr(prompt_client, "name", request.prompt_name),
            prompt_label=request.prompt_label,
            prompt_version=getattr(prompt_client, "version", request.prompt_version),
            prompt_type=request.prompt_type,
            compiled_text=compiled_text.strip(),
            messages=messages,
            variables=variables,
            is_fallback=bool(getattr(prompt_client, "is_fallback", False)),
            fingerprint=_fingerprint(compiled_text),
        )

    @staticmethod
    def _normalize_chat_message(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            if message.get("type") == "placeholder":
                return {
                    "role": "placeholder",
                    "content": f"{{{{{message.get('name', 'placeholder')}}}}}",
                }
            return {
                "role": str(message.get("role", "unknown")),
                "content": str(message.get("content", "")),
            }
        return {"role": "unknown", "content": str(message)}

    @staticmethod
    def _stringify_chat_messages(messages: list[dict[str, Any]]) -> str:
        lines = [f"[{message.get('role', 'unknown')}] {message.get('content', '')}" for message in messages]
        return "\n".join(lines).strip()


_DEFAULT_PROMPT_RESOLVER: PromptResolverService | None = None


def _fingerprint(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]



def _get_default_prompt_resolver() -> PromptResolverService:
    global _DEFAULT_PROMPT_RESOLVER
    if _DEFAULT_PROMPT_RESOLVER is None:
        _DEFAULT_PROMPT_RESOLVER = PromptResolverService(LangfuseCollectorClient())
    return _DEFAULT_PROMPT_RESOLVER



def resolve_prompt(request: PromptResolutionRequest) -> PromptResolutionResult:
    return _get_default_prompt_resolver().resolve_prompt(request)


def list_prompts(
    *,
    name: str | None = None,
    label: str | None = None,
    tag: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return _get_default_prompt_resolver().list_prompts(name=name, label=label, tag=tag, limit=limit)
