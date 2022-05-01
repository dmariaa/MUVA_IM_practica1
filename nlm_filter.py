import numpy as np
import scipy.ndimage.filters as filters


def slides(data: np.ndarray, ws: int = 3, p: int = 1) -> np.ndarray:
    """
    Generates windowed view of input array, for a window size and a padding
    :param data: input array
    :param ws: window size
    :param p: padding
    :return: windowed view of input array
    """
    rs = (data.shape[0] - ws + 2 * p) + 1
    cs = (data.shape[1] - ws + 2 * p) + 1

    data = np.pad(data, ((p, p), (p, p)))
    shape = (rs, cs, ws, ws)
    strides = (data.strides[0], data.strides[1], data.strides[0], data.strides[1])
    s = np.lib.stride_tricks.as_strided(data, shape, strides)

    return s


def nlm(im: np.ndarray, ws: int, h: float):
    padding = ws // 2
    slides_im = slides(im, ws, padding)
    result = np.zeros(im.shape)

    for i in np.arange(im.shape[0]):
        for j in np.arange(im.shape[1]):

            current = slides_im[i, j]
            distances = np.exp(-(np.sqrt(np.sum((slides_im - current) ** 2, axis=(2, 3))) / (h ** 2)))
            z = np.sum(distances)
            w = distances / z
            result[i, j] = np.sum(w * im)

    return result


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import cv2.cv2 as cv2
    from noise import add_noise, NoiseTypes


    def read_image(file):
        image = cv2.imread(file, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float64)
        image = image / 255.
        return image

    image = read_image("materiales/T1.png")
    noise_image = add_noise(image=image, noise_type=NoiseTypes.RICIAN_NOISE, intensity=0.1)
    start = time.time()
    denoised_image = nlm(im=noise_image, ws=3, h=0.5)
    print(f"Elapsed: {time.time() - start:.4f} seconds")

    plt.interactive(True)
    plt.imshow(image, cmap='gray')
    plt.title("original")
    plt.show()

    plt.imshow(noise_image, cmap='gray')
    plt.title("ruidosa")
    plt.show()

    plt.imshow(denoised_image, cmap='gray')
    plt.title("denoised")
    plt.show()
