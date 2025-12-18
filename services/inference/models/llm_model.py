from omegaconf import OmegaConf
from pathlib import Path


class LLMModel:
    def __init__(self, model_name: str):
        self.cls = self._get_model_class(model_name)
        self.cfg = self._get_model_config(model_name)
        self.model = self.cls(self.cfg)
        self.model.load_model()

    def _get_model_class(self, model_name: str):
        if model_name == "qwen2-5_instruct":
            from .qwen_modeling import Qwen2_5Instruct
            return Qwen2_5Instruct
        else:
            raise ValueError(f"Модель {model_name} не поддерживается")
    
    def _get_model_config(self, model_name: str):
        cfg_name = f"{model_name}_config.yaml"
        cfg_path = Path(__file__).parent / f"model_configs/{cfg_name}"
        cfg = OmegaConf.load(cfg_path)
        return cfg
    
    def get_response(self, batch: list[str]):
        requests = self.model.generate_response(batch)
        return requests