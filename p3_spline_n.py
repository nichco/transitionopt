import csdl
import numpy as np

class p3_B(csdl.Model):
    def initialize(self):
        self.parameters.declare('N')
    def define(self):
        N = self.parameters['N'] # number of control points
        n = N - 1 # number of spline segments
        
        p3c = self.declare_variable('p3c', shape=(N,))
        y = csdl.expand(p3c, (N,1), 'ij->iajb')
        
        B3 = self.create_output('B3', val=0, shape=(4*n,1))
        
        # type A: curves meet at same point
        for j in range(1,n-1+1):
            B3[2*j-1-1, 0] = 1*y[j+1-1, 0]
            B3[2*j-1, 0] = 1*y[j+1-1, 0]
        
        # type D: specified endpoints
        B3[2*(n-1)+(n-1)+(n-1)+1-1, 0] = 1*y[1-1, 0]
        B3[2*(n-1)+(n-1)+(n-1)+2-1, 0] = 1*y[n+1-1, 0]

class p3_cubic(csdl.Model):
    def initialize(self):
        self.parameters.declare('N')
    def define(self):
        N = self.parameters['N'] # number of control points
        n = N - 1 # number of splines
        
        A_inv = self.declare_variable('A_inv',shape=(4*n,4*n))
        B3 = self.declare_variable('B3',shape=(4*n,1))

        coef3 = csdl.matmat(A_inv,B3)
        self.register_output('coef3', coef3)
        
class p3_interpolate(csdl.Model):
    def initialize(self):
        self.parameters.declare('num')
        self.parameters.declare('N')
    def define(self):
        num = self.parameters['num']
        N = self.parameters['N'] # number of control points
        n = N - 1 # number of splines
        x = np.linspace(0, num, N)
        
        coef3 = self.declare_variable('coef3',shape=(4*n,1))
        
        # interpolation
        p3 = self.create_output('p3', val=0, shape=(num,1))
        for i in range(n):
            a = coef3[4*i, 0]
            b = coef3[4*i+1, 0]
            c = coef3[4*i+2, 0]
            d = coef3[4*i+3, 0]
            xi = x[i]
            xf = x[i+1]
            for j in range(int(xi), int(xf)):
                p3[j,0] = a + b*j + c*j**2 + d*j**3
        
        # integration
        p3_integral = self.create_output('p3_integral', val=0, shape=(n,1))
        for i in range(n):
            a = coef3[4*i, 0]
            b = coef3[4*i+1, 0]
            c = coef3[4*i+2, 0]
            d = coef3[4*i+3, 0]
            xi = x[i]
            xf = x[i+1]
            p3_integral[i,0] = (a*xf + (1/2)*b*xf**2 + (1/3)*c*xf**3 + (1/4)*d*xf**4) - (a*xi + (1/2)*b*xi**2 + (1/3)*c*xi**3 + (1/4)*d*xi**4)


