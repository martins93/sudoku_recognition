import numpy as np
import cv2
from matplotlib import pyplot as plt


# Cargar Imagen
image_sudoku_original = cv2.imread('/home/martin/sudoku/sudoku_recognition/sudoku-original.jpg')
# Mostrar Imagen

cv2.imshow("Imagen original",image_sudoku_original)
cv2.waitKey(0)


img = cv2.GaussianBlur(image_sudoku_original,(5,5),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros((gray.shape),np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

#Performs advanced morphological transformations.
#http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=cv2.morphologyex#cv2.morphologyEx
close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
div = np.float32(gray)/(close)
res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

cv2.imshow("Imagen morfologia",res2)
cv2.waitKey(0)


# Conversion de imagen de su espacio de colores a escala de grises
#image_sudoku_gray = cv2.cvtColor(image_sudoku_original, cv2.COLOR_BGR2GRAY)
# Binarizar imagen por umbral (adaptive threshold)
#gray = cv2.GaussianBlur(img, (5, 5), 0)
#thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)[1]

thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)

#cv2.imshow("Imagen original",image_sudoku_original)
#cv2.waitKey(0)

# show image
cv2.imshow("Imagen binarizada",thresh)
cv2.waitKey(0)

# El metodo findContours encuentra los contornos en una imagen binarizada
# El vector hierarchy indica el contorno hijo o padre de contours0, en caso de no tener
# hijo o padre hierarchy es negativo
# CV_RETR_TREES retrieves all of the contours and reconstructs a full hierarchy of nested contours.
# CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
# For example, an up-right rectangular contour is encoded with 4 points.

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Size of the image (height, width)

# copy the original image to show the posible candidate
image_sudoku_candidates = image_sudoku_original.copy()

# We are taking a practical assumption :
# The biggest square in the image should be Sudoku Square.
# In short, image should be taken close to Sudoku.

size_rectangle_max = 0;
biggest = None
max_area = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 100:
        peri = cv2.arcLength(i, True)
        approximation = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approximation) == 4:
            biggest = approximation
            max_area = area

# show the best candidate
for i in range(len(approximation)):
    cv2.line(thresh,
             (biggest[(i % 4)][0][0], biggest[(i % 4)][0][1]),
             (biggest[((i + 1) % 4)][0][0], biggest[((i + 1) % 4)][0][1]),
             (255, 0, 0), 2)
# show image
cv2.imshow("Imagen contorno",thresh)
cv2.waitKey(0)

# Cambio de perspectiva de la imagen

def rectify(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


approx = rectify(biggest)
h = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)

retval = cv2.getPerspectiveTransform(approx, h)
warp_gray = cv2.warpPerspective(gray, retval, (450, 450))


#cv2.imshow("Imagen perspectiva",test)
#cv2.waitKey(0)

# Binarizar imagen por umbral (adaptive threshold)
# var1 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
# var2 = cv2.GaussianBlur(gray, (5, 5), 0


#var2 = cv2.adaptiveThreshold(var1, 255, 1, 1, 11, 2)


cv2.imshow("Imagen perspectiva",warp_gray)
cv2.waitKey(0)

var1 = cv2.GaussianBlur(warp_gray, (5, 5), 0)
var2 = cv2.adaptiveThreshold(var1,255,0,1,19,2)

cv2.imshow("Imagen ultimo",var2)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(var2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



squares = []
size_rectangle_max = 0;
biggest = None
max_area = 0
h=0
w=0
for i in contours:
   area = cv2.contourArea(i)
   if area > 100:
       peri = cv2.arcLength(i, True)
       if peri<200 and peri>150:
           approximation = cv2.approxPolyDP(i, 0.04 * peri, True)
           if len(approximation) == 4:
               squares.append(approximation)
               #area = cv2.contourArea(approximation)
               #print("AREA: "+str(area))
               #print(cv2.moments(approximation))
               #x, y, w, h = cv2.boundingRect(approximation)
               #print("DATA: "+str(x)+str(y)+str(w)+str(h))

for i in range(len(squares)):
    e = squares[i]
    for j,obj in enumerate(e):
        if j == 3:
            cv2.line(var2,(e[j][0][0], e[j][0][1]),(e[0][0][0], e[0][0][1]),(255, 0, 1), 2)
            break
        else:
            cv2.line(var2, (e[j][0][0], e[j][0][1]), (e[j + 1][0][0], e[j + 1][0][1]), (255, 0, 1), 2)


h, w = warp_gray.shape[:2]
cv2.imshow("Imagen perspectiva",var2)
cv2.waitKey(0)


im_number = warp_gray[0*52:(0+1)*52][:, 3*49:(3+1)*49]


var1 = cv2.GaussianBlur(im_number, (5, 5), 0)



var2 = cv2.adaptiveThreshold(var1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
  cv2.THRESH_BINARY,11,2)


cv2.imshow('Output', var2)
cv2.waitKey(0)
cv2.imwrite('/home/martin/sudoku/sudoku_recognition/var2.png',var2)




