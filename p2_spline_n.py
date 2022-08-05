import csdl
import numpy as np

class p2_B(csdl.Model):
    def initialize(self):
        self.parameters.declare('N')
    def define(self):
        N = self.parameters['N'] # number of control points
        n = N - 1 # number of spline segments
        
        p2c = self.declare_variable('p2c', shape=(N,))
        y = csdl.expand(p2c, (N,1), 'ij->iajb')
        
        B2 = self.create_output('B2', val=0, shape=(4*n,1))
        
        # type A: curves meet at same point
        for j in range(1,n-1+1):
            B2[2*j-1-1, 0] = 1*y[j+1-1, 0]
            B2[2*j-1, 0] = 1*y[j+1-1, 0]
        
        # type D: specified endpoints
        B2[2*(n-1)+(n-1)+(n-1)+1-1, 0] = 1*y[1-1, 0]
        B2[2*(n-1)+(n-1)+(n-1)+2-1, 0] = 1*y[n+1-1, 0]

class p2_cubic(csdl.Model):
    def initialize(self):
        self.parameters.declare('N')
    def define(self):
        N = self.parameters['N'] # number of control points
        n = N - 1 # number of splines
        
        A_inv = self.declare_variable('A_inv',shape=(4*n,4*n))
        B2 = self.declare_variable('B2',shape=(4*n,1))

        coef2 = csdl.matmat(A_inv,B2)
        self.register_output('coef2', coef2)
        
class p2_interpolate(csdl.Model):
    def initialize(self):
        self.parameters.declare('num')
        self.parameters.declare('N')
    def define(self):
        num = self.parameters['num']
        N = self.parameters['N'] # number of control points
        n = N - 1 # number of splines
        x = np.linspace(0, num, N)
        
        coef2 = self.declare_variable('coef2',shape=(4*n,1))
        
        # interpolation
        p2 = self.create_output('p2', val=0, shape=(num,1))
        for i in range(n):
            a = coef2[4*i, 0]
            b = coef2[4*i+1, 0]
            c = coef2[4*i+2, 0]
            d = coef2[4*i+3, 0]
            xi = x[i]
            xf = x[i+1]
            for j in range(int(xi), int(xf)):
                p2[j,0] = a + b*j + c*j**2 + d*j**3
        
        # integration
        p2_integral = self.create_output('p2_integral', val=0, shape=(n,1))
        for i in range(n):
            a = coef2[4*i, 0]
            b = coef2[4*i+1, 0]
            c = coef2[4*i+2, 0]
            d = coef2[4*i+3, 0]
            xi = x[i]
            xf = x[i+1]
            p2_integral[i,0] = (a*xf + (1/2)*b*xf**2 + (1/3)*c*xf**3 + (1/4)*d*xf**4) - (a*xi + (1/2)*b*xi**2 + (1/3)*c*xi**3 + (1/4)*d*xi**4)


