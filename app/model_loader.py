import torch
from transformers import AutoModelForImageSegmentation


class ModelLoader:

    _model = None

    @classmethod
    def load(cls):

        if cls._model is None:

            model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                trust_remote_code=True
            )

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            model = model.float()

            model.to(device)
            model.eval()

            cls._model = model

        return cls._model