from pprint import pprint
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import cast

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
        """
        Fill the template with both static fields (from config)
        and dynamic fields (from example).
        """
        context = {**self.fields, **example}
        return self.template.format(**context)

