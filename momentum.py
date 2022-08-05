# Glauert Momentum Model
import csdl
import csdl_om
import numpy as np


class Solve_System(csdl.Model):
    def initialize(self):
        self.parameters.declare('a')
        self.parameters.declare('k')
        self.parameters.declare('rho')
        self.parameters.declare('p_str', types=str)
        self.parameters.declare('v_str', types=str)
        
    def define(self):
        a = self.parameters['a']
        k = self.parameters['k']
        rho = self.parameters['rho']
        p_str = self.parameters['p_str']
        v_str = self.parameters['v_str']
        
        v_axial = self.declare_variable(v_str)
        p = self.declare_variable(p_str)
        
        t = self.declare_variable('t')
        r = t*v_axial + k*t*((-v_axial/2) + (((v_axial**2)/4) + (t/(2*rho*a)))**0.5) - p
        self.register_output('r', r)

class Glauert(csdl.Model):
    def initialize(self):
        self.parameters.declare('a')
        self.parameters.declare('k')
        self.parameters.declare('rho')
        self.parameters.declare('p_str', types=str)
        self.parameters.declare('v_str', types=str)
        self.parameters.declare('out_name', types=str)
        
    def define(self):
        a = self.parameters['a']
        k = self.parameters['k']
        rho = self.parameters['rho']
        p_str = self.parameters['p_str']
        v_str = self.parameters['v_str']
        out_name = self.parameters['out_name']
        
        v_axial = self.declare_variable(v_str)
        p = self.declare_variable(p_str)
        
        solver = self.create_implicit_operation(Solve_System(a=a, k=k, rho=rho, p_str=p_str, v_str=v_str))
        solver.declare_state('t', residual='r', bracket=(-20000, 20000))
        """
        solver.declare_state('t', residual='r')
        
        solver.nonlinear_solver = csdl.NewtonSolver(
          solve_subsystems=False,
          maxiter=200,
          iprint=False,
        )
        solver.linear_solver = csdl.ScipyKrylov()
        """
        thrust = solver(v_axial, p)
        self.register_output(out_name, 1*thrust)
        





# for testing
class Run(csdl.Model):
    def define(self):
        
        r = 3 # m
        a = np.pi*r**2
        k = 1.2
        rho = 1.225
        
        self.create_input('p', val=300000)
        self.create_input('u', val=40)
        
        p_str = 'p'
        v_str = 'u'
        
        self.add(Glauert(a=a, k=k, rho=rho, p_str=p_str, v_str=v_str, out_name='thrust'))


# for testing
sim = csdl_om.Simulator(Run())
sim.run()
print(sim['thrust'])






