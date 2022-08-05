import csdl
import numpy as np

class p1_B(csdl.Model):
    def initialize(self):
        self.parameters.declare('N')
    def define(self):
        N = self.parameters['N'] # number of control points
        n = N - 1 # number of spline segments
        
        p1c = self.declare_variable('p1c', shape=(N,))
        y = csdl.expand(p1c, (N,1), 'ij->iajb')
        
        B1 = self.create_output('B1', val=0, shape=(4*n,1))
        
        # type A: curves meet at same point
        for j in range(1,n-1+1):
            B1[2*j-1-1, 0] = 1*y[j+1-1, 0]
            B1[2*j-1, 0] = 1*y[j+1-1, 0]
        
        # type D: specified endpoints
        B1[2*(n-1)+(n-1)+(n-1)+1-1, 0] = 1*y[1-1, 0]
        B1[2*(n-1)+(n-1)+(n-1)+2-1, 0] = 1*y[n+1-1, 0]

class p1_cubic(csdl.Model):
    def initialize(self):
        self.parameters.declare('N')
    def define(self):
        N = self.parameters['N'] # number of control points
        n = N - 1 # number of splines
         
        A_inv = self.declare_variable('A_inv',shape=(4*n,4*n)) 
        B1 = self.declare_variable('B1',shape=(4*n,1))

        coef1 = csdl.matmat(A_inv,B1)
        self.register_output('coef1', coef1)
        
class p1_interpolate(csdl.Model):
    def initialize(self):
        self.parameters.declare('num')
        self.parameters.declare('N')
    def define(self):
        num = self.parameters['num']
        N = self.parameters['N'] # number of control points
        n = N - 1 # number of splines
        x = np.linspace(0, num, N)
        
        coef1 = self.declare_variable('coef1',shape=(4*n,1))
        
        # interpolation
        p1 = self.create_output('p1', val=0, shape=(num,1))
        for i in range(n):
            a = coef1[4*i, 0]
            b = coef1[4*i+1, 0]
            c = coef1[4*i+2, 0]
            d = coef1[4*i+3, 0]
            xi = x[i]
            xf = x[i+1]
            for j in range(int(xi), int(xf)):
                p1[j,0] = a + b*j + c*j**2 + d*j**3
        
        # integration
        p1_integral = self.create_output('p1_integral', val=0, shape=(n,1))
        for i in range(n):
            a = coef1[4*i, 0]
            b = coef1[4*i+1, 0]
            c = coef1[4*i+2, 0]
            d = coef1[4*i+3, 0]
            xi = x[i]
            xf = x[i+1]
            p1_integral[i,0] = (a*xf + (1/2)*b*xf**2 + (1/3)*c*xf**3 + (1/4)*d*xf**4) - (a*xi + (1/2)*b*xi**2 + (1/3)*c*xi**3 + (1/4)*d*xi**4)


