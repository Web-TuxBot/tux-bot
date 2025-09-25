from omegaconf import DictConfig
import hydra
from models.hf_model import HFModel

# TODO (mortiferr): Реализовать асинхронный инференс модели и написать ручку
@hydra.main(version_base=None, config_path="configs", config_name="config")
def inference(cfg: DictConfig):
    qwen = HFModel(cfg=cfg)
    while True:
        print(f"Ваш запрос: ", end='')
        prompt = input()
        answer= qwen.generate(prompt,
                              max_new_tokens=cfg.model_settings.max_new_tokens,
                              temperature=cfg.model_settings.temperature,
                              top_p=cfg.model_settings.top_p)
        print(f"Ответ чат-бота: {answer}\n")  

if __name__ == "__main__":
    inference()