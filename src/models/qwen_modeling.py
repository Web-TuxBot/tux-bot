from abc import abstractmethod, ABC
import time
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from omegaconf import DictConfig

class Qwen2_5Model(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_name_or_path = self.cfg.loading.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate_response(self, **kwargs):
        pass

class Qwen2_5Instruct(Qwen2_5Model):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            dtype=self.cfg.loading.param_type,
            device_map=self.cfg.loading.device_map,
            low_cpu_mem_usage=self.cfg.loading.low_cpu_mem_usage
        )
    
    def save_model(self, save_dir: str, sharded_mem: str | None = None):
        if sharded_mem is None:
            self.model.save_pretrained(f"{save_dir}/model")
        else:
            self.model.save_pretrained(f"{save_dir}/model", max_shard_size=sharded_mem)
        self.tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    
    @torch.no_grad()
    def generate_response(self, request: str):
        messages = [
            {"role": "system", "content": self.cfg.model.system_prompt},
            {"role": "user", "content": request}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_genereation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.cfg.model.max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
