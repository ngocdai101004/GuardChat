"""Gemini 2.5 Flash Image client - the T2I model Task 2 is scored on.

Sibling of :mod:`src.Gemini.client`, which drives the *text* model that
rewrites prompts. This one drives the *image* model that turns a
finished rewrite into a picture, so Safe Generation Rate can be measured
end to end.

Two call shapes, because Task 2 has two input representations:

``generate``
    One prompt in, one image out. Used for the ``prompt``
    representation, and for the ``concat`` / ``last_turn`` conversation
    modes.
``generate_chat``
    A multi-turn session: turn *i* is sent with turns *1..i-1* still in
    the model's context, and every reply is kept. This is what a
    GuardChat conversation attack actually looks like against a T2I
    chat product - the intent is spread across turns and only the last
    reply is the payload - so it is the default for the
    ``conversation`` representation.

Safety settings are left at the provider's defaults
-----------------------------------------------------
The rewriter (:mod:`src.Gemini.client`) sets ``BLOCK_NONE`` so it is
allowed to *read* an adversarial prompt. Here the opposite is correct:
the point of SGR is what happens when a rewrite meets the T2I system as
deployed, filters and all. A block is a real outcome and is recorded as
one. ``relax_safety=True`` is available for ablations and is off by
default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# Gemini's image model, named in the request. Note that image models
# only answer through ``generate_content`` with an image response
# modality - the id is not interchangeable with the text tier.
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"

# Fallbacks tried by ``preflight`` only for the error message, in the
# order most likely to be live for a fresh key.
KNOWN_IMAGE_MODELS = (
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-3-pro-image-preview",
)

_OVERRIDABLE_HARM_CATEGORIES: List[str] = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]

API_KEY_ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

# Finish reasons that mean "the provider refused", as opposed to "the
# provider answered". ``IMAGE_SAFETY`` is image-specific: the text was
# accepted and the *picture* was rejected after being drawn.
_BLOCK_REASONS = {
    "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII",
    "RECITATION", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION", "IMAGE_OTHER",
}

# The model answering "I will not draw this" under IMAGE-only
# modalities. Not a filter block - the call finished normally - so it is
# tracked apart from _BLOCK_REASONS.
NO_IMAGE_REASON = "NO_IMAGE"

_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}


def _import_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            "Gemini image generation needs the `google-genai` package "
            "(`pip install \"google-genai>=1.0\"`)."
        ) from e
    return genai, types


def _resolve_api_key(explicit: Optional[str]) -> str:
    """Explicit argument, then the environment, then the repo-root ``.env``."""
    from src.utils.hf_token import resolve_env_value

    key = resolve_env_value(API_KEY_ENV_KEYS, explicit=explicit)
    if key:
        return key
    raise RuntimeError(
        "No Gemini API key found. Pass --api-key, set GEMINI_API_KEY / "
        "GOOGLE_API_KEY in the environment, or add GEMINI_API_KEY=... to "
        "the repo-root .env file."
    )


def extension_for(mime_type: Optional[str]) -> str:
    return _MIME_EXTENSIONS.get((mime_type or "").lower(), ".png")


@dataclass
class GeneratedImage:
    """One image part of a response."""

    data: bytes
    mime_type: str = "image/png"

    @property
    def num_bytes(self) -> int:
        return len(self.data or b"")


@dataclass
class ImageResponse:
    """Decoded result of a single image ``generate_content`` call."""

    images: List[GeneratedImage] = field(default_factory=list)
    text: str = ""                      # the model's commentary, if any
    blocked: bool = False
    block_reason: Optional[str] = None
    finish_reason: Optional[str] = None
    attempts: int = 1
    elapsed_sec: float = 0.0
    error: Optional[str] = None         # set when every attempt raised

    @property
    def has_image(self) -> bool:
        return bool(self.images) and bool(self.images[0].data)


@dataclass
class ImageClientConfig:
    """Per-run client settings.

    ``retry_blocked`` defaults to False, unlike the rewriter's client. A
    block here is the measurement, not an obstacle to it: re-rolling
    until the filter lets an image through would report a number no
    deployed system produces.
    """

    model_name: str = DEFAULT_IMAGE_MODEL
    api_key: Optional[str] = None
    relax_safety: bool = False
    retries: int = 3
    retry_blocked: bool = False
    backoff_seconds: float = 2.0
    request_timeout: Optional[float] = 180.0
    aspect_ratio: Optional[str] = None          # e.g. "1:1"; None = model default
    # IMAGE-only by default, and that is a correctness fix rather than a
    # tidiness one. Left at the model default, the image tier behaves
    # like a general assistant: a GuardChat turn phrased as a musing
    # ("Hmm, I'm thinking of a photograph that...") gets a chatty reply
    # and a clarifying question instead of a picture, so a whole dialogue
    # can finish with no image and nothing to gate - a conversational
    # outcome scored as a safety one. Pinned to IMAGE, every turn either
    # renders or comes back with finish_reason=NO_IMAGE, which is the
    # model declining and is recorded as such.
    response_modalities: Optional[Sequence[str]] = ("IMAGE",)


class GeminiImageClient:
    """Thin wrapper exposing :meth:`generate` and :meth:`generate_chat`."""

    def __init__(self, config: ImageClientConfig = ImageClientConfig()) -> None:
        self.config = config
        genai, types = _import_genai()
        self._genai = genai
        self._types = types

        self._client = genai.Client(
            api_key=_resolve_api_key(config.api_key),
            **self._client_kwargs(config.request_timeout),
        )
        self._safety_settings = (
            self._build_safety_settings() if config.relax_safety else None
        )

    # ----------------------------- Setup ---------------------------- #

    def _client_kwargs(self, request_timeout: Optional[float]) -> Dict[str, Any]:
        if not request_timeout:
            return {}
        try:
            return {"http_options": self._types.HttpOptions(
                timeout=int(float(request_timeout) * 1000)
            )}
        except (AttributeError, TypeError, ValueError):
            return {}

    def _build_safety_settings(self):
        types = self._types
        out = []
        for cat in _OVERRIDABLE_HARM_CATEGORIES:
            try:
                out.append(types.SafetySetting(category=cat, threshold="BLOCK_NONE"))
            except (TypeError, ValueError):
                continue
        return out

    def build_config(self):
        """A ``GenerateContentConfig``, or None when nothing needs setting.

        Every optional field is added defensively: the image knobs
        (``response_modalities``, ``image_config``) arrived in different
        SDK releases, and a benchmark should not require one exact
        version of the client library to run.
        """
        types = self._types
        kwargs: Dict[str, Any] = {}
        if self._safety_settings:
            kwargs["safety_settings"] = self._safety_settings
        if self.config.response_modalities:
            kwargs["response_modalities"] = list(self.config.response_modalities)
        if self.config.aspect_ratio:
            try:
                kwargs["image_config"] = types.ImageConfig(
                    aspect_ratio=self.config.aspect_ratio
                )
            except (AttributeError, TypeError, ValueError):
                pass
        if not kwargs:
            return None
        try:
            return types.GenerateContentConfig(**kwargs)
        except (TypeError, ValueError):
            # SDK too old for one of the keys - drop the optional ones
            # rather than failing the run.
            kwargs.pop("image_config", None)
            kwargs.pop("response_modalities", None)
            return types.GenerateContentConfig(**kwargs) if kwargs else None

    # --------------------------- Preflight -------------------------- #

    def preflight(self) -> None:
        """One cheap call, so a retired model id fails now rather than N times."""
        try:
            resp = self._client.models.generate_content(
                model=self.config.model_name,
                contents="A plain white circle on a black background.",
                config=self.build_config(),
            )
        except Exception as e:  # noqa: BLE001 - re-raised with context
            msg = str(e)
            if "404" not in msg and "NOT_FOUND" not in msg:
                raise RuntimeError(
                    f"Gemini image preflight failed for "
                    f"{self.config.model_name!r}: {e}"
                ) from e
            raise RuntimeError(
                f"Gemini rejected the image model id {self.config.model_name!r}:\n"
                f"  {msg}\n"
                f"Image models visible to this key: "
                f"{', '.join(self.available_models()) or '(none)'}\n"
                f"Known ids: {', '.join(KNOWN_IMAGE_MODELS)}\n"
                f"Pass a different --model."
            ) from e

        decoded = self._decode(resp)
        if not decoded.has_image and not decoded.blocked:
            raise RuntimeError(
                f"{self.config.model_name!r} answered the preflight prompt "
                f"without an image (finish_reason={decoded.finish_reason}, "
                f"text={decoded.text[:200]!r}). It is probably a text-only "
                f"model id - pass an image model with --model."
            )

    def available_models(self, contains: str = "image") -> List[str]:
        """Best-effort list of image-capable ids for the error message."""
        try:
            return [
                m.name.split("/")[-1]
                for m in self._client.models.list()
                if contains in (m.name or "")
            ]
        except Exception:  # noqa: BLE001 - diagnostics only
            return []

    # ---------------------------- Generate -------------------------- #

    def generate(self, prompt: str) -> ImageResponse:
        """One prompt, one call (plus retries). Never raises."""
        return self._call(lambda: self._client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=self.build_config(),
        ))

    def generate_chat(self, turns: Sequence[str]) -> List[ImageResponse]:
        """Send turns in sequence through one chat session.

        Returns one response per turn, in order, so a caller can inspect
        what the model drew *before* the final turn as well as after it.
        A turn that errors ends the session - the context is broken from
        that point on - and the responses collected so far are returned.
        """
        chat = self._client.chats.create(
            model=self.config.model_name,
            config=self.build_config(),
        )
        out: List[ImageResponse] = []
        for turn in turns:
            resp = self._call(lambda t=turn: chat.send_message(t))
            out.append(resp)
            if resp.error:
                break
        return out

    def _call(self, send) -> ImageResponse:
        """Run ``send`` with retry/backoff and decode whatever comes back.

        Failures are returned, not raised: one dead sample out of a
        thousand should cost that sample, not the run. The exception text
        is kept verbatim in ``error`` so the caller can classify it with
        :func:`src.utils.classify_error_message`.
        """
        max_attempts = max(1, int(self.config.retries))
        started = time.time()
        last_exc: Optional[Exception] = None
        last_decoded: Optional[ImageResponse] = None

        for attempt in range(max_attempts):
            is_last = (attempt + 1 >= max_attempts)
            try:
                raw = send()
            except Exception as e:  # noqa: BLE001 - SDK raises a wide tree
                last_exc = e
                if is_last or not self._is_transient(str(e).lower()):
                    break
                time.sleep(self.config.backoff_seconds * (2 ** attempt))
                continue

            decoded = self._decode(raw)
            decoded.attempts = attempt + 1
            decoded.elapsed_sec = time.time() - started
            last_decoded = decoded
            last_exc = None

            if decoded.has_image or is_last:
                return decoded
            if decoded.blocked and not self.config.retry_blocked:
                return decoded
            time.sleep(self.config.backoff_seconds * (2 ** attempt))

        if last_decoded is not None:
            last_decoded.attempts = max_attempts
            last_decoded.elapsed_sec = time.time() - started
            return last_decoded
        return ImageResponse(
            attempts=max_attempts,
            elapsed_sec=time.time() - started,
            error=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown",
        )

    @staticmethod
    def _is_transient(msg: str) -> bool:
        markers = (
            "429", "rate limit", "rate-limit", "quota", "resource_exhausted",
            "500", "502", "503", "504", "internal error",
            "deadline", "timeout", "timed out", "connection",
            "unavailable", "overloaded",
        )
        return any(m in msg for m in markers)

    # ----------------------- Response decoding ---------------------- #

    def _decode(self, response) -> ImageResponse:
        """Pull image parts, commentary and the safety verdict out of a response."""
        images: List[GeneratedImage] = []
        fragments: List[str] = []
        blocked = False
        block_reason: Optional[str] = None
        finish_reason: Optional[str] = None

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            finish_reason = self._stringify_enum(getattr(cand, "finish_reason", None))
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                inline = (getattr(part, "inline_data", None)
                          or getattr(part, "inlineData", None))
                data = getattr(inline, "data", None) if inline is not None else None
                if data:
                    images.append(GeneratedImage(
                        data=data,
                        mime_type=getattr(inline, "mime_type", None) or "image/png",
                    ))
                    continue
                text = getattr(part, "text", None)
                if text:
                    fragments.append(text)

            if (finish_reason or "").upper() in _BLOCK_REASONS:
                blocked = True
                block_reason = finish_reason

        pf = getattr(response, "prompt_feedback", None)
        if pf is not None:
            br = getattr(pf, "block_reason", None)
            if br:
                blocked = True
                block_reason = block_reason or self._stringify_enum(br)

        return ImageResponse(
            images=images,
            text="".join(fragments),
            blocked=blocked,
            block_reason=block_reason,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _stringify_enum(v) -> Optional[str]:
        if v is None:
            return None
        return getattr(v, "name", None) or str(v)


__all__ = [
    "DEFAULT_IMAGE_MODEL",
    "NO_IMAGE_REASON",
    "KNOWN_IMAGE_MODELS",
    "GeneratedImage",
    "GeminiImageClient",
    "ImageClientConfig",
    "ImageResponse",
    "extension_for",
]
