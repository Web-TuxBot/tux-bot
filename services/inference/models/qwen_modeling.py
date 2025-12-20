from abc import abstractmethod, ABC
from omegaconf import DictConfig
import hydra
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from omegaconf import DictConfig

class Qwen2_5Model(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_name_or_path = self.cfg.loading.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left")
    
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
    def generate_response(self, batch: list[str]):
        messages = [
            [
                {"role": "system", "content": self.cfg.model.system_prompt},
                {"role": "user", "content": request}   
            ]
            for request in batch
        ]
        texts = [
            self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_genereation_prompt=True
            )
            for message in messages 
        ]

        model_inputs = self.tokenizer(texts, 
                                      return_tensors="pt",
                                      padding=True, 
                                      truncation=True).to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.cfg.model.max_new_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        processed_responses = []
        for text in responses:
            words = text.strip().split(maxsplit=1)

            # если вдруг модель выдала 0 или 1 слово — не ломаемся
            if len(words) < 2:
                processed_responses.append(text)
                continue

            first_char_upper = words[1][0].upper() + words[1][1:]
            processed_responses.append(first_char_upper)

        return processed_responses