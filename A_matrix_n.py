# function to create the universal A matrix for natural cubic spline calculations
import numpy as np


def A_matrix(num,y):
    N = len(y) # number of control points
    x = np.linspace(0, num, N)
    n = N - 1 # number of spline segments
    
    A = np.zeros((4*n,4*n))
    
    # type A: curves meet at same point
    for j in range(1,n-1+1):
        A[2*j-1-1,(4*j-3-1):(4*j)] = np.array([1,x[j+1-1],x[j+1-1]**2,x[j+1-1]**3])
        A[2*j-1,(4*j+1-1):(4*j+4)] = np.array([1,x[j+1-1],x[j+1-1]**2,x[j+1-1]**3])
    
    # type B: slopes match at meeting points
    for k in range(1,n-1+1):
        A[2*(n-1)+k-1,(4*k-3-1):(4*k+4)] = np.array([0,1,2*x[k+1-1],3*x[k+1-1]**2,0,-1,-2*x[k+1-1],-3*x[k+1-1]**2])
    
    # type C: curvatures match at meeting points
    for m in range(1,n-1+1):
        A[2*(n-1)+(n-1)+m-1,(4*m-3-1):(4*m+4)] = np.array([0,0,2,6*x[m+1-1],0,0,-2,-6*x[m+1-1]])
        
    # type D: specified endpoints
    A[2*(n-1)+(n-1)+(n-1)+1-1,(1-1):4] = np.array([1,x[1-1],x[1-1]**2,x[1-1]**3])
    A[2*(n-1)+(n-1)+(n-1)+2-1,(4*n-3-1):(4*n)] = np.array([1,x[n+1-1],x[n+1-1]**2,x[n+1-1]**3])
    
    # type E: curvature at endpoints = 0
    A[2*(n-1)+(n-1)+(n-1)+3-1,(1-1):4] = np.array([0,0,2,6*x[1-1]])
    A[2*(n-1)+(n-1)+(n-1)+4-1,(4*n-3-1):(4*n)] = np.array([0,0,2,6*x[n+1-1]])
    
    A_inv = np.linalg.inv(A)

    return A, A_inv