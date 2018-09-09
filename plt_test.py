# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

args = sys.argv

if len(args) != 2 :
	print ("\t(!) args != 2")
	sys.exit()

# epoch , train_mean_loss , train_acc , test_acc
plt.figure(num=None, figsize=(12,8), dpi=80, facecolor='w', edgecolor='k')
files = np.sort(os.listdir(args[1]))
mark  = ["-","--","-s","-o","-d","-p","--x","-->","--<"]


# plot *.csv
for i in range(len(files)) :
	if files[i].endswith('.csv') == True :
		data = np.loadtxt((args[1] + files[i]),delimiter=",")
		plt.plot(data[:,0],data[:,3],mark[i],label=files[i])
		print (files[i]," : max ",np.max(data[:,3]))

x = np.array([i for i in range(51)])
y = np.array([i for i in range(51)])
y = y.astype(np.float32)

border = 0.95
for i in range(51) :	
	y[i] = border
# print (x)
# print (y)
plt.plot(x,y,'-',color="#000000")
plt.text(-1.8,border,str(border),color="#000000",ha='left',va='center')

plt.legend(loc='lower right')
plt.xlabel("epoch")
plt.ylabel("test_accuracy")
plt.grid(True)

plt.ylim(0.0, 1.0)
outname = args[1] + 'testacc.eps'
plt.savefig(outname)
plt.show()
