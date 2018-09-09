# coding:utf-8
# pickle化したデータを再度画像に復元して，データの正しさを確認する

import numpy as np
import cv2
import os

import pickle
f = open('test.dat', 'rb')
mnist = pickle.load(f)
f.close()

for num in range(360):
    img = mnist['data'][num].reshape(80,123)#(20name)(80,123)#(split)(110,47)#(113,47)
    img = img*255
    filename = str(num)+'.png'
    cv2.imwrite('tmp/'+filename,img)


'''
img = cv2.resize(img, (dimX,dimY))
deleteDegree(img, dimX, dimY)
cv2.imshow('image', img)
#cv2.imwrite('img.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
