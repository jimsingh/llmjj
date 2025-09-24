import yaml
import string
from pathlib import Path

class PromptBuilder:
    def __init__(self, base_cfg: dict, use_cfg: dict, overrides: dict):
        """
        Args:
            base_cfg: The base.yaml config (template + fragments).
            use_cfg: The eval.yaml that maps which fragments to use.
            overrides: The eval.yaml["overrides"]
        """
        self.template = base_cfg["base"]["template"]
        self.fragments = base_cfg["fragments"]
        self.use_cfg = use_cfg
        self.overrides = overrides or {}

    @classmethod
    def from_files(cls, base_yaml_path: str, eval_yaml_path: str):
        base_cfg = cls._load_yaml(base_yaml_path)
        eval_cfg = cls._load_yaml(eval_yaml_path)
        return cls(base_cfg, eval_cfg["use"], eval_cfg.get("overrides"))

    @staticmethod
    def _load_yaml(path: str) -> dict:
        with Path(path).open("r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _find_placeholders(template_str: str) -> set[str]:
        formatter = string.Formatter()
        return {field for _, field, _, _ in formatter.parse(template_str) if field}

    def _fill_template(self, template_str: str, **kwargs) -> str:
        required = self._find_placeholders(template_str)
        missing = required - kwargs.keys()
        if missing:
            raise KeyError(f"unresolved template params: {missing}")
        return template_str.format(**kwargs)

    def _resolve_fragments(self) -> dict:
        resolved = {}
        for key, option in self.use_cfg.items():
            fragment_value = self.fragments[key][option]
            if isinstance(fragment_value, dict):
                resolved[key] = yaml.dump(fragment_value, default_flow_style=False).strip()
            else:
                resolved[key] = fragment_value.format(**self.overrides)
        return resolved

    def render(self) -> str:
        resolved = self._resolve_fragments()
        params = {**resolved, **self.overrides}
        return self._fill_template(self.template, **params)
