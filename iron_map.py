import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def normalize_img(img):
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    norm_result = (img_norm * 255).astype(np.uint8)
    return norm_result

def main():

    # Establecer formato de los argumentos para llamada del programa
    parser = argparse.ArgumentParser(description='Cuantificación de hierro en MRI')
    parser.add_argument('-d', '-directory', '--directory', required=True, type=str,
                        help='Especifique la ruta en la que se encuentran las imágenes que desea procesar')
    parser.add_argument('-c', '-case', '--case', required=True, type=str,
                        help='Indique el caso (paciente) que desea procesar')

    args = vars(parser.parse_args())

    directory = args['directory']
    case = args['case']

    print('Iniciando procesamiento. Por favor, espere unos instantes')
    mri_names = [(img_name) for img_name in os.listdir(directory) if img_name.startswith(case)]
    TE = [(int(mri_names[i].split('TE')[1].split('.')[0])) for i in range(len(mri_names))]
    names_sorted = [name for _, name in sorted(zip(TE, mri_names))]
    TE_sorted = sorted(TE)
    mri_imgs = [(cv2.imread(directory + mri, 0)) for mri in names_sorted]
    h, w = mri_imgs[0].shape
    mri_imgs = [cv2.copyMakeBorder(mri, h - mri.shape[0], 0, w - mri.shape[1], 0, cv2.BORDER_CONSTANT)
                for mri in mri_imgs]

    R2 = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            y = np.array([(mri_imgs[l][i, j]) for l in range(len(mri_imgs))])
            y_refined = np.delete(y, np.where(y == 0))
            x_refined = np.delete(TE_sorted, np.where(y == 0))
            if len(y_refined) >= 2:
                log_y = np.log(y_refined, dtype=np.float64)
                B, ln_A = np.polyfit(x_refined, log_y, 1, w = np.sqrt(y_refined))
                R2[i, j] = - B


    R2_norm = normalize_img(R2)
    T2 = np.zeros((h, w))
    T2[R2_norm != 0] = 1 / R2_norm[R2_norm != 0]
    T2_norm = (T2 * 255).astype(np.uint8)

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(R2_norm, cmap='gray')
    plt.colorbar()
    plt.title('Mapa R2 (normalizado)', color='#1453ff', size=20)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(R2_norm, cmap='jet')
    plt.colorbar()
    plt.title('R2 (mapa de color "jet")', color='#1453ff', size=20)

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(T2_norm, cmap='gray')
    plt.title('T2 (normalizado)', color='#1453ff', size=20)
    plt.colorbar()

    plt.show()


    return print('El procesamiento ha finalizado.')

if __name__ == "__main__":
    main()



