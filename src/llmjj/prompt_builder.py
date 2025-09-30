from pprint import pprint
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import cast, Tuple

class PromptBuilder:
    def __init__(self, config_path: str):
        cfg = OmegaConf.load(config_path)
        base_cfg = OmegaConf.load("conf/base.yaml")

        self.cfg: DictConfig = cast(DictConfig, OmegaConf.merge(base_cfg, cfg))

        self.template: str = self.cfg.get("template", "")
        self.fields = self.cfg.get("fields", {})

        print(f"loaded template {config_path}:")
        pprint(OmegaConf.to_container(self.cfg, resolve=True))
        

    def build(self, example: dict) -> str:
        context = {**self.fields, **example}
        return self.template.format(**context)

    def parse_response(self, raw_output: str) -> Tuple[str, str]:
        """Convert the raw LLM output into a reason and a label"""
        clean = raw_output.strip().strip('|')
        parts = [s.strip().strip('"') for s in clean.split('|') if s.strip()]
        return parts[0], parts[-1]
