from enum import Enum

import numpy as np


class NoiseTypes(Enum):
    RICIAN_NOISE = 1
    GAUSSIAN_NOISE = 2


def add_noise(image: np.ndarray, noise_type: NoiseTypes, **kwargs) -> np.ndarray:
    return_image = image.copy()

    if noise_type == NoiseTypes.RICIAN_NOISE:
        """
        Inspired by https://stackoverflow.com/questions/67006926/how-do-you-add-rician-noise-to-an-image
        """
        intensity = kwargs.get('intensity', 0.2)

        v = 1
        s = 1
        N = image.size

        noise = np.random.normal(scale=s, size=(N, 2)) + [[v, 0]]
        noise = np.linalg.norm(noise, axis=1)

        mean_noise = np.mean(noise)
        mean_image = np.mean(image)
        factor = intensity * mean_image / mean_noise
        return_image += (factor * noise).reshape(image.shape)

    return return_image
