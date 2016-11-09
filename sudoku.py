import numpy as np
import cv2
import math
from sudoku_recognition import predict


image_sudoku_original = cv2.imread('/home/martin/sudoku/sudoku_recognition/sudoku-hard.jpg')

cv2.imshow("Imagen original",image_sudoku_original)
cv2.waitKey(0)


img = cv2.GaussianBlur(image_sudoku_original,(5,5),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("Imagen en escala de grises",gray)
cv2.waitKey(0)

thresh1 = cv2.adaptiveThreshold(gray,255,0,1,19,2)


cv2.imshow("Imagen binarizada",thresh1)
cv2.waitKey(0)



contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#image_sudoku_candidates = image_sudoku_original.copy()

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

for i in range(len(approximation)):
    cv2.line(image_sudoku_original,
             (biggest[(i % 4)][0][0], biggest[(i % 4)][0][1]),
             (biggest[((i + 1) % 4)][0][0], biggest[((i + 1) % 4)][0][1]),
             (255, 0, 0), 2)

cv2.imshow("Contorno principal",image_sudoku_original)
cv2.waitKey(0)

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


h, w = warp_gray.shape[:2]
print h
print w

cv2.imshow("Imagen con cambio perspectiva",warp_gray)
cv2.waitKey(0)


var2 = cv2.adaptiveThreshold(warp_gray,255,0,1,19,2)
#close = cv2.morphologyEx(var2,cv2.MORPH_CLOSE,kernel1)



gauss = cv2.GaussianBlur(warp_gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(gauss,255,0,1,19,2)


kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(thresh, kernel, iterations=1)
dilation = cv2.dilate(thresh, kernel, iterations=1)


closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

#
# close = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel1)
# div = np.float32(warp_gray)/(close)
# res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
# res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)


#img = cv2.GaussianBlur(var2,(5,5),0)


cv2.imshow("Imagen con Closing",closing)
cv2.waitKey(0)

# cv2.imshow("Imagen ultimo",thresh)
# cv2.waitKey(0)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def nroCuadrado(x, y,w,h):

    width = 449
    height = 449

    x = x+(w/2)
    y = y+(h/2)

    widthxCuadrado = width / 9
    heightxCuadrado = height / 9

    for i in range(0, 9):
        for j in range(0, 9):
            proximoenAncho = (i + 1) * widthxCuadrado
            actualenAncho = i * widthxCuadrado
            proximoenAlto = (j + 1) * heightxCuadrado
            actualenAlto = j * heightxCuadrado
            if (x >=  actualenAncho and x <= proximoenAncho and y >=  actualenAlto and y <= proximoenAlto):
                return i, j

sudoku_matrix = np.zeros((9,9))
squares = []
size_rectangle_max = 0;
biggest = None
max_area = 0
count = 0
area_total = 0
for i in contours:
   area = cv2.contourArea(i)
   if area > 100:
           approximation = cv2.approxPolyDP(i, 0.04 * peri, True)
           if len(approximation) == 4:
                   area = cv2.contourArea(approximation)
                   if area > 1000 and area <=3000:
                       squares.append(approximation)
                       area = cv2.contourArea(approximation)
                       area_total += area
                       count +=1

                       x, y, w, h = cv2.boundingRect(approximation)
                       #print("X: "+str(x)+" Y: "+str(y)+" W: "+str(w)+ " H: "+str(h))
                       cv2.rectangle(gauss, (x, y), (x + w, y + h), (0, 255, 0), 2)
                       new_image = gauss[x+7:x+h-7, y+7:y+w-7]

                       f, g = nroCuadrado(x, y,w,h)
                       print (f, g)
                       #
                       var2 = cv2.adaptiveThreshold(new_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                     cv2.THRESH_BINARY, 11, 2)

                       non_black = cv2.countNonZero(var2)
                       total = var2.size

                       percent = (float(non_black)/float(total))*100

                       if percent > 90.0:
                           number = -1
                       else:
                           number = predict.main(var2)
                           #var = 1



                       print ("EL NUMERO ES: " + str(number))
                       sudoku_matrix[f][g] = number
                       print(sudoku_matrix)


                       #print(number)

                       #cv2.imshow("Imagen perspectiva", var2)
                       #cv2.waitKey(0)
                       #name = '/home/lcorniglione/Documents/sudoku_recognition/fotos/var%s%d.jpg' %(f,g)

                       #cv2.imwrite(name, var2)




result = (area_total/count)
area_prom = math.sqrt(result)
print (sudoku_matrix)


print ("CANTIDAD RECONOCIDA:")
print (len(squares))

for i in range(len(squares)):
    e = squares[i]
    for j,obj in enumerate(e):
        if j == 3:
            cv2.line(warp_gray,(e[j][0][0], e[j][0][1]),(e[0][0][0], e[0][0][1]),(255, 0, 1), 2)
            break
        else:
            cv2.line(warp_gray, (e[j][0][0], e[j][0][1]), (e[j + 1][0][0], e[j + 1][0][1]), (255, 0, 1), 2)


h, w = warp_gray.shape[:2]
cv2.imshow("Imagen cuadrados",warp_gray)
cv2.waitKey(0)


im_number = warp_gray[0*(area_prom+8.5):(0+1)*(area_prom+8.5)][:, 2*(area_prom+8.5):(2+1)*(area_prom+8.5)]



var1 = cv2.GaussianBlur(im_number, (5, 5), 0)
var2 = cv2.adaptiveThreshold(var1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
  cv2.THRESH_BINARY,11,2)



cv2.imshow('Output', var2)
cv2.waitKey(0)
cv2.imwrite('/home/martin/sudoku/sudoku_recognition/var2.png',var2)


