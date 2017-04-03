import os
import random
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# note we could have used EVD because A.T A is a symmetric pos def matrix 
# from this we could concieve V.T(Sig**2)V and then found U to make the ald much faster

markers = [u'D', u'o',u's',u'^',u'x',u'*',u'p',u'H',u'+',u'.']
colors = ['r','g','b','y','m','c','k','b','k','r']
image_path = "C:\\Users\\niles\\Desktop\hw2\\faces_jpg\\"
file_path = "C:\\Users\\niles\\Desktop\hw2\\princomps.eps"


folder = 'faces_jpg' # the images folder
m = 84*69            # image size
n = 400              # number of images
A = np.zeros((m, n), dtype=np.float64, order='F')

# TODO
# Step 1: Read image files and store them into A
def get_data(Arr): 
	for index,file in enumerate(os.listdir(folder)):
		Arr[index] =  misc.imread(image_path + file).reshape(m)
	return Arr.T

C = np.zeros((n,m),dtype=np.float64)
A = get_data(C)
# TODO
# Step 2: Perform PCA on the images
# first calculate mean then subtract from each data point in the col
# our features are in the cols for each image...
# center the data of matrix A making PCA=SVD
means = np.mean(A,axis=1).reshape(m,1)
A -= means

# perform the svd function now on the data now
# svd takes the data as cols so we are already good to use it
U, s, V_t = np.linalg.svd(A, full_matrices=False) 
PC = V_t.T*s 
# could use
# np.dot(np.diag(s),V_t) which would be must slower

# this matrix is now composed of rows that are the principal components 
# cols which are data points

# TODO
# Step 3: Generate the first five eigenfaces 

for image in xrange(5):
	misc.imsave("u" + str(image+1)+".jpg", U.T[image].reshape([84,69]))

# TODO
# Step 4: Randomly pick 10 persons and plot their first two principal components
# arrays to be used for plotting
rand_img = []
# x = []
# y = []
# checks for the cluster to have at least have one data point
count = 0
while count < 10:
	rand_index = random.randint(0,39) #randint(a,b) chooses x s.t a <= x <= b
	if rand_index not in rand_img:
		rand_img.append(rand_index)
		start = rand_index*10
		end  = 10*(rand_index+1)
		#x.append(PC[0][start:end]) <-- uneeded parts of code.... line 73 works all into one 
		#y.append(PC[1][start:end])
		m = markers[count]
		c = colors[count]	
		plt.scatter(PC[0][start:end], PC[1][start:end], marker=m, color=c)
		count+=1 
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
# check if the file is open and if so delete it
if os.path.isfile(file_path):
	os.remove(file_path)
plt.savefig(file_path, format = 'eps', dpi=1000)
plt.show()