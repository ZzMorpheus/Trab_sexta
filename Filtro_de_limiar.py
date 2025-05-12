import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem em escala de cinza
img = cv2.imread('imagem.jpg', cv2.IMREAD_GRAYSCALE)

# 1. FILTRO DE INVERSÃO
img_invertida = 255 - img

# 2. FILTRO DE LIMIAR (THRESHOLD)
limiar = 127  # valor de corte
_, img_limiar = cv2.threshold(img, limiar, 255, cv2.THRESH_BINARY)

# Mostrar resultados
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagem Original')

plt.subplot(1, 3, 2)
plt.imshow(img_invertida, cmap='gray')
plt.title('Filtro de Inversão')

plt.subplot(1, 3, 3)
plt.imshow(img_limiar, cmap='gray')
plt.title('Filtro de Limiar')

plt.tight_layout()
plt.show()
