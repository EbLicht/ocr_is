# coding: utf-8
# chainer用に名前データを生成する．
# srcDir以下のディレクトリ内の画像をdatファイルにする

import numpy as np
import cv2
import os

np.random.seed(0)
srcDir      = 'jikken3_person/jikken3_persontrain46/'
outfilename = 'jikken3_person/jikken3_persontrain46.dat'

# ℃の文字を255でpaddingして消す．
def deleteDegree(img, dimX, dimY):
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
#            if (int(dimX*0.75)<x and int(dimY*0.55)<y):
#                img[y,x] = np.uint8(255)
            
            # ついでに2値化
            if img[y,x]>127:
                img[y,x]=np.uint8(255)
            else:
                img[y,x]=np.uint8(0)

user = os.listdir(srcDir)

# 各ユーザの顔画像の枚数が違うため，画像数をカウントする
print ("[directory,images]")
image_num = 0
for u in user:
    if u!='.DS_Store':
        n = len(os.listdir(srcDir+str(u))) # - 1
        print("	",u, n)
        image_num += n
print('	(image_num =', image_num,")")    

# 画像のサイズ
dimX = 123
dimY = 80
dim = dimX * dimY

data   = np.zeros(image_num * dim, dtype=np.float32).reshape((image_num, dim))
target = np.zeros(image_num, dtype=np.int32).reshape((image_num, ))

print ("image -> data")
ct = 0
for (i, u) in enumerate(user):
    print("	",i,u)
    if u!='.DS_Store':

        for (j, f) in enumerate(os.listdir(srcDir + u + '/')):
                 #if f!='._.DS_Store':
                 text = srcDir + u + '/' + f
                 print (text)
                 img = cv2.imread(srcDir + u + '/' + f, cv2.IMREAD_GRAYSCALE)
                 img = cv2.resize(img, (dimX,dimY))
                 deleteDegree(img, dimX, dimY)
                 img = img / 255
                 trg = img
                 img = [flatten for inner in img for flatten in inner]
                 for c in range(dim):
                        data[ct,c] = img[c]
                        target[ct] = (int(f[-6])*5+int(f[-5]))%20 
                 ct += 1


# ランダムシャッフル
print ("Shuffle data")
data2 = np.zeros(image_num * dim, dtype=np.float32).reshape((image_num, dim))
target2 = np.zeros(image_num, dtype=np.int32).reshape((image_num, ))

import random
indexlist = list(range(ct))
random.shuffle(indexlist)

for i in range(ct):
    for c in range(dim):
        data2[i, c] = data[indexlist[i], c]
    target2[i] = target[indexlist[i]]

print ("data -> file")
mnist = {}
mnist['data'] = data2
mnist['target'] = target2

import pickle
f = open(outfilename, 'wb')
pickle.dump(mnist, f)
f.close()

'''
img = cv2.imread('temperature/fujita/mark_fujita6_54413.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (dimX,dimY))
deleteDegree(img, dimX, dimY)
cv2.imshow('image', img)
#cv2.imwrite('img.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
