import numpy as np
import cv2.cv2 as cv2
import matplotlib.pyplot as plt


def dif_aniso(im: np.ndarray, niter: int, k: float, l: float, option: int) -> np.ndarray:
    assert im.ndim == 2, "Image must be gray (2 channel)"

    rows, cols = im.shape
    imdif = im.copy()

    for i in range(niter):
        # pads image with zeros
        imdifm = np.pad(imdif, (1,), constant_values=0, mode='constant')

        # gradients in 4 directions
        deltaN = imdifm[0:rows, 1:cols + 1] - imdif
        deltaS = imdifm[2:rows + 2, 1:cols + 1] - imdif
        deltaE = imdifm[1:rows + 1, 2:cols + 2] - imdif
        deltaW = imdifm[1:rows + 1, 0:cols] - imdif

        # conductance
        if option == 1:
            cN = np.exp(-(deltaN / k) ** 2)
            cS = np.exp(-(deltaS / k) ** 2)
            cE = np.exp(-(deltaE / k) ** 2)
            cW = np.exp(-(deltaW / k) ** 2)
        elif option == 2:
            cN = 1. / np.exp(1. + (deltaN / k) ** 2)
            cS = 1. / np.exp(1. + (deltaS / k) ** 2)
            cE = 1. / np.exp(1. + (deltaE / k) ** 2)
            cW = 1. / np.exp(1. + (deltaW / k) ** 2)

        imdif = imdif + (l * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW))
        pass

    return imdif


if __name__ == "__main__":
    from noise import add_noise, NoiseTypes

    image = cv2.imread("materiales/T1.png", cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float64)
    plt.imshow(image, cmap='gray')
    plt.title("original")
    plt.show()

    noise_image = add_noise(image=image, noise_type=NoiseTypes.RICIAN_NOISE, intensity=0.3)
    plt.imshow(noise_image, cmap='gray')
    plt.title("ruidosa")
    plt.show()

    denoised_image = dif_aniso(im=noise_image, niter=5, k=40, l=0.25, option=2)
    plt.imshow(denoised_image, cmap='gray')
    plt.title("denoised")
    plt.show()
