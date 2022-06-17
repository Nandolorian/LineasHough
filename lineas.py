import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# Se lee y se muestra la imagen a procesar para detectar las líneas.
img = cv2.imread('./lineas.png')
plt.figure(facecolor="#E8E4E4",figsize=(6, 6)).canvas.set_window_title("Imagen a procesar")
plt.imshow(img)
plt.show()

# Se remueve el color de la imagen
imgGris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Se hace un blur para mejorar la detección de bordes
imgBlur = cv2.GaussianBlur(imgGris, (5, 5), 1.5)
bordes = cv2.Canny(imgBlur, 100, 200)

# Se muestra la imagen procesada
plt.figure(facecolor="#E8E4E4",figsize=(6, 6)).canvas.set_window_title("Bordes detectados")
plt.imshow(bordes)
plt.show()

# Tomamos las medidas de la imagen
rango_y, rango_x = bordes.shape
diagonal = int(math.hypot(rango_x,rango_y)+1)

# Calculamos los rangos de Rho y Theta
rango_rho = np.arange(-diagonal, diagonal + 1, 1)
rango_theta = np.deg2rad(np.arange(-90, 90, 1))

# Creamos el acumulador usando los rangos antes calculados y lo inicializamos en cero
acumulador = np.zeros((len(rango_rho), len(rango_theta)), dtype=np.uint64)

# Buscamos las bordes en la imagen para procesarlos
bordes_y, bordes_x = np.nonzero(bordes) 
for i in range(len(bordes_x)):
    x = bordes_x[i]
    y = bordes_y[i]
    # Por cada posible theta calculamos el rho
    for j in range(len(rango_theta)):
        rho = int((x * np.cos(rango_theta[j]) + y * np.sin(rango_theta[j])) + diagonal)
        # Actualizamos la celda del acumulador
        acumulador[rho, j] += 1

# Buscamos los índices de los máximos dentro del acumulador usando un límite para máximos locales
limite = 6
# Asignamos los índices de los máximos locales
indices =  np.argpartition(acumulador.flatten(), -2)[-limite:]
# Generamos las lineas detectadas
lineas = np.vstack(np.unravel_index(indices, acumulador.shape)).T

# Graficamos el espacio de Hough para visualizar los máximos
plt.figure(facecolor="#E8E4E4",figsize=(10, 10)).canvas.set_window_title("Espacio de Hough")
plt.imshow(acumulador, cmap='turbo')
plt.show()

# Graficamos las líneas detectadas
for i in range(len(lineas)):
    rho = rango_rho[lineas[i][0]]
    theta = rango_theta[lineas[i][1]]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

plt.figure(facecolor="#E8E4E4",figsize=(6, 6)).canvas.set_window_title("Lineas detectadas")
plt.imshow(img, cmap='turbo')
plt.show()
