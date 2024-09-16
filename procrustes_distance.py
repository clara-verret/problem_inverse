import numpy as np
from numpy import sqrt, transpose, sum, trace
from numpy.linalg import norm, svd

list_landmarks_1 = [(0,1),(0,2),(1,2),(1,1)]#[(0,0),(0,1),(1,0)]
list_landmarks_2 = [(3,1),(4,2),(3,3),(2,2)]#[(0,1),(1,0),(2,1)]#[(3,1),(2,2),(3,3),(4,2)]
k = len(list_landmarks_1)

X_1 = np.array(list_landmarks_1)
X_2 = np.array(list_landmarks_2)

H = np.zeros((k-1,k))

for i in range(k-1):
	for j in range(i+1):
		H[i,j] = -1/sqrt((i+1)*(i+2))
	H[i,i+1] = (i+1)/sqrt((i+1)*(i+2))
	
Z_1 = H @ X_1
Z_1 = Z_1/norm(Z_1)
Z_2 = H @ X_2
Z_2 = Z_2/norm(Z_2)

U, S, Vh = svd(Z_2.transpose() @ Z_1)

d_procustes = sqrt(1 - sum(S)**2)
print(d_procustes)

d_sibson = trace(X_1.transpose() @ X_1) + trace(X_2.transpose() @ X_2) - 2*sqrt(trace(sqrt(X_1.transpose() @ X_1) @ (X_2.transpose() @ X_2) @ sqrt(X_1.transpose() @ X_1)))
print(d_sibson)
