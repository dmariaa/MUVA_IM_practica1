import numpy as np
import scipy.ndimage.filters as filters


def nlm(image, ws, h, p):
    w = ws // 2
    nlm_filter = 0
    z = 0

    image0 = image[p - w:p + 1 + w, p - w:p + 1 + w]

    for i in range(-w, w + 1):
        for j in range(-w, w + 1):
            # región definida
            ngbr_pixel = image[p + i - w:p + 1 + i + w, p + j - w:p + 1 + j + w]
            d_euc = np.sqrt((image0 - ngbr_pixel) ** 2)  # distancia euclídea
            z += np.exp(-d_euc / (h ** 2))

    # Cálculo de Z
    Z = np.sum(z)

    # Ecuación final del filtro NLM
    nlm_filter = (1 / Z) * z
    nlm_filter = np.ones(nlm_filter.shape, np.uint8)
    nlm_filter = nlm_filter * nlm_filter
    nlm_filter_final = nlm_filter / Z

    # Convolución entre la imagen y la expresión final del filtro NLM
    image_filtered = filters.convolve(image, nlm_filter_final, mode='reflect')

    return image_filtered  # Devuelve la imagen filtrada
