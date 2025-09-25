from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from omegaconf import DictConfig

class HFModel:
    def __init__(self, cfg: DictConfig):
        self.model_path = cfg.model.model_path
        self.tokenizer_path = cfg.model.tokenizer_path

        self.system_prompt = cfg.model_settings.system_prompt
        self.device_map = cfg.model_settings.device_map
        self.max_seq_len = cfg.model_settings.max_seq_len
        self.param_type = cfg.model_settings.param_type
        self._from_pretrained()

    def _from_pretrained(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=self.param_type,
            #device_map=self.device_map
            device_map="cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 4096, temperature: float = 0.7, top_p: float = 0.9):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

        outputs = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]