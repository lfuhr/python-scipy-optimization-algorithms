import numpy as np


H = np.eye(3)*2
A = np.matrix('1 2 -1; 1 -1 1')
g = np.matrix('0;0;0')
b = np.matrix('-4;2')

n = A.shape[1]
m = A.shape[0]

# print('\nA =', A)

block = np.block([[H, np.transpose(A)],
		  		[A, np.zeros((2,2))]])

x = np.linalg.solve(block, np.block([[-g],[-b]]))
print('\nx =', x)

# Nebenbedingungen testen
print('\nSolver p =', x[0:3])
print('\nA*p = -b =', A*x[0:3])


QT, LSchlangeT  = np.linalg.qr(A.T, mode="complete") # q*r = A
Q1T = QT[0:n, 0:m]
Q2T = QT[0:n, m:n]
Q2 = Q2T.T
LT = LSchlangeT[0:m,0:m]
L = LT.T
LInv = L.I
y1 = - LInv * b

y2 = (Q2*H*Q2T).I * -Q2 * (g - H * Q1T * LInv * b)

p = Q1T * y1 + Q2T * y2

print('\nNullraum p =', p)
