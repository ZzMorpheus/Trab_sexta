import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem em escala de cinza
img = cv2.imread('imagem.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar o filtro Laplaciano
laplaciano = cv2.Laplacian(img, cv2.CV_64F)  # CV_64F evita problemas com números negativos
laplaciano = cv2.convertScaleAbs(laplaciano)  # Converte para uint8 (imagem exibível)

# Exibir imagem original e com filtro
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagem Original")

plt.subplot(1, 2, 2)
plt.imshow(laplaciano, cmap='gray')
plt.title("Filtro Laplaciano (Realce de Bordas)")

plt.tight_layout()
plt.show()
