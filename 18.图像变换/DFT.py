import cv2
import numpy as np

img = cv2.imread('../img/1.png',0)
rows,cols = img.shape
print(rows,cols)
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
print(nrows,ncols)
nimg = np.zeros((nrows,ncols))
nimg[:rows,:cols] = img
print(nimg.shape)