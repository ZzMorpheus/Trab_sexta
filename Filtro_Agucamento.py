import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem colorida
img = cv2.imread('imagem.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Definir um kernel de aguçamento simples (realce de bordas)
kernel_aguçamento = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])

# Aplicar o filtro usando convolução
img_aguçada = cv2.filter2D(img_rgb, -1, kernel_aguçamento)

# Mostrar as imagens
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Imagem Original')

plt.subplot(1, 2, 2)
plt.imshow(img_aguçada)
plt.title('Imagem com Aguçamento')

plt.tight_layout()
plt.show()
