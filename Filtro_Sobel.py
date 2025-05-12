import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem em escala de cinza
img = cv2.imread('imagem.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar filtro de Sobel nas direções X e Y
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Derivada em X
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Derivada em Y

# Converter para escala de 0 a 255
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combinar as duas direções (magnitude do gradiente)
sobel_total = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Exibir imagens
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagem Original")

plt.subplot(1, 4, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title("Sobel X (Horizontal)")

plt.subplot(1, 4, 3)
plt.imshow(sobel_y, cmap='gray')
plt.title("Sobel Y (Vertical)")

plt.subplot(1, 4, 4)
plt.imshow(sobel_total, cmap='gray')
plt.title("Sobel Total (Bordas)")

plt.tight_layout()
plt.show()
