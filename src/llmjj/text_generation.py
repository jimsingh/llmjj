from dataclasses import dataclass
import platform
from typing import Any, Protocol
from transformers import pipeline

class TextGenerator(Protocol):
    """ using structural typing (Protocol) instead of ABC, class below don't need to derive""" 
    def generate(self, prompt: str, *, max_new_tokens: int, truncation: bool = True, **kwargs: Any) -> str:
        ...

    def __call__(self, prompt: str, *, max_new_tokens: int, truncation: bool = True, **kwargs: Any) -> str:
        ...


class HuggingFaceGenerator:

    def __post_init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = pipeline("text-generation", model=self.model_name, return_full_text=False)

    def generate(self, prompt: str, *, max_new_tokens: int, truncation: bool = True, **kwargs: Any) -> str:
        options: dict[str, Any] = {"max_new_tokens": max_new_tokens, "truncation": truncation}
        options.update(kwargs)
        result = self._pipeline(prompt, **options)
        return result[0]["generated_text"]

    __call__ = generate


class MLXGenerator:

    def __init__(self, model_name: str) -> None:
        from mlx_lm import load
        self.model_name = model_name
        self._model, self._tokenizer = load(self.model_name)

    def generate(self, prompt: str, *, max_new_tokens: int, truncation: bool = True, **kwargs: Any) -> str:
        from mlx_lm import generate as mlx_generate

        options: dict[str, Any] = {"max_tokens": max_new_tokens}
        options.update(kwargs)
        completion = mlx_generate(self._model, self._tokenizer, prompt=prompt, **options)
        return completion 

    __call__ = generate


def build_text_generator(model_name: str) -> TextGenerator:
    if platform.system().lower() == "darwin":
        return MLXGenerator(model_name)
    return HuggingFaceGenerator(model_name)
