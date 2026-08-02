"""Task-2 Safe Generation Rate: text-to-image + the image-safety gate.

Two halves, deliberately kept separable:

:mod:`~src.ImageGen.client`
    Gemini 2.5 Flash Image, the T2I model a rewrite is tested against.
:mod:`~src.ImageGen.classifier`
    The ResNet-152 gate ported from Image-Generation-Guardian, which
    decides whether a generated image counts as safe.

:mod:`~src.ImageGen.pipeline` joins them into the SGR loop and
:mod:`~src.ImageGen.generate` is the CLI. Swapping in another T2I
backend (FLUX.1, DALL-E 3) means writing another client with
``generate`` / ``generate_chat``; nothing else moves.
"""

from .classifier import (
    CATEGORIES as IMAGE_CATEGORIES,
    SAFE_CATEGORY,
    ImageSafetyClassifier,
    resolve_weights,
)
from .client import (
    DEFAULT_IMAGE_MODEL,
    GeminiImageClient,
    GeneratedImage,
    ImageClientConfig,
    ImageResponse,
)
from .pipeline import (
    CONVERSATION_MODES,
    VERDICTS,
    apply_verdicts,
    STATUSES as GENERATION_STATUSES,
    TEXT_FIELDS,
    GenerationPipeline,
    GenerationRecord,
    summarise,
    verdict_for,
)

__all__ = [
    "CONVERSATION_MODES",
    "DEFAULT_IMAGE_MODEL",
    "GENERATION_STATUSES",
    "IMAGE_CATEGORIES",
    "SAFE_CATEGORY",
    "TEXT_FIELDS",
    "VERDICTS",
    "apply_verdicts",
    "GeminiImageClient",
    "GeneratedImage",
    "GenerationPipeline",
    "GenerationRecord",
    "ImageClientConfig",
    "ImageResponse",
    "ImageSafetyClassifier",
    "resolve_weights",
    "summarise",
    "verdict_for",
]
