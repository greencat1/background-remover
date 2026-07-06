from PIL import Image
import numpy as np


def postprocess(mask, image):

    # убираем лишние измерения
    mask = np.squeeze(mask)

    # нормализуем
    mask = (mask - mask.min()) / (
        mask.max() - mask.min() + 1e-8
    )

    # в uint8
    mask = (mask * 255).astype(
        np.uint8
    )

    alpha = Image.fromarray(
        mask
    )

    # приводим маску к размеру исходного изображения
    alpha = alpha.resize(
        image.size,
        Image.Resampling.LANCZOS
    )

    # переводим изображение в RGBA
    image = image.convert(
        "RGBA"
    )

    image.putalpha(
        alpha
    )

    return image