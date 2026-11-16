"""Project policy atop DeerFlow's existing DeepSeek adapter."""
from deerflow.models.patched_deepseek import PatchedChatDeepSeek

from .model_policy import ScopedModelMixin


class ScopedDeepSeek(ScopedModelMixin, PatchedChatDeepSeek):
    pass
