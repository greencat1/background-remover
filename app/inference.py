from io import BytesIO
import time

import torch
import numpy as np

from PIL import Image

from app.model_loader import ModelLoader
from app.preprocessing import preprocess
from app.postprocessing import postprocess


class BackgroundService:

    def __init__(self):

        print("[INIT] Loading model...")

        self.model = ModelLoader.load()

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[INIT] Model loaded on {self.device}"
        )


    def remove_background(
        self,
        image_bytes
    ):

        total_start = time.time()

        print("\n========================")
        print("[REQUEST] New image received")

        step = time.time()

        image = Image.open(
            BytesIO(image_bytes)
        )

        original = image.copy()

        print(
            f"[LOAD] Image size: {image.size}"
        )

        print(
            f"[LOAD] Time: {time.time()-step:.2f}s"
        )


        step = time.time()

        tensor = preprocess(
            image
        )

        tensor = tensor.to(
            self.device
        )

        print(
            f"[PREPROCESS] Tensor shape: {tensor.shape}"
        )

        print(
            f"[PREPROCESS] Time: {time.time()-step:.2f}s"
        )


        step = time.time()

        print(
            "[INFERENCE] Running model..."
        )

        with torch.no_grad():

            prediction = self.model(
                tensor
            )

        print(
            f"[INFERENCE] Time: {time.time()-step:.2f}s"
        )


        step = time.time()

        print(
            "[POSTPROCESS] Processing mask..."
        )

        mask = torch.sigmoid(
            prediction[-1]
        )

        mask = mask.cpu()

        mask = mask.numpy()

        mask = np.squeeze(
            mask
        )

        result = postprocess(
            mask,
            original
        )

        print(
            f"[POSTPROCESS] Time: {time.time()-step:.2f}s"
        )


        step = time.time()

        output = BytesIO()

        result.save(
            output,
            format="PNG"
        )

        output.seek(0)

        print(
            f"[SAVE] Time: {time.time()-step:.2f}s"
        )

        print(
            f"[TOTAL] Total request time: {time.time()-total_start:.2f}s"
        )

        print("========================\n")

        return output