import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para aplicar uma curva de tons a cada canal da imagem
def aplicar_curva(img, curva):
    # Aplica a curva a cada canal (B, G, R)
    return cv2.LUT(img, curva)

# Funções de curvas
def curva_linear():
    return np.array([i for i in range(256)], dtype='uint8')

def curva_invertida():
    return np.array([255 - i for i in range(256)], dtype='uint8')

def curva_s():
    # Curva em S (mais contraste)
    return np.array([np.clip(0.5 * (i - 128) + 128 + 0.5 * (i - 128)**3 / (128**2), 0, 255) for i in range(256)], dtype='uint8')

# Carregar imagem colorida
img = cv2.imread('imagem.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Aplicar curvas
curva_orig = curva_linear()
curva_inv = curva_invertida()
curva_s_shape = curva_s()

img_original = aplicar_curva(img, curva_orig)
img_invertida = aplicar_curva(img, curva_inv)
img_contraste = aplicar_curva(img, curva_s_shape)

# Exibir resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_original)
plt.title("Curva Linear (Original)")

plt.subplot(1, 3, 2)
plt.imshow(img_contraste)
plt.title("Curva em S (Contraste)")

plt.subplot(1, 3, 3)
plt.imshow(img_invertida)
plt.title("Curva Invertida (Negativo)")

plt.tight_layout()
plt.show()
