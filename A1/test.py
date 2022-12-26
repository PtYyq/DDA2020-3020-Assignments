import numpy as np

import A1_120090761 as grading
X = np.array([[1,2],[4,3],[5,6],[3,8],[9,10]])
y = np.array([[-1],[0],[1],[0],[0]])
w, XT, InvXTX = grading.A1_120090761(X,y)
print(w)
print(XT)
print(InvXTX)