from ozone.api import ODEProblem, Wrap, NativeSystem
import openmdao.api as om
import csdl
import csdl_om
import numpy as np
import matplotlib.pyplot as plt
import time

from ode import ODESystemModel
from t_vec import timestep
from A_matrix_n import A_matrix

from p1_spline_n import p1_cubic, p1_interpolate, p1_B
from p2_spline_n import p2_cubic, p2_interpolate, p2_B
from p3_spline_n import p3_cubic, p3_interpolate, p3_B

from power import Power
from transition_efficiency import Transition_Efficiency
"""
from modopt.csdl_library import CSDLProblem
from modopt.scipy_library import SLSQP
from modopt.optimization_algorithms import SQP
from modopt.snopt_library import SNOPT
"""
# ODE problem class
class ODEProblemTest(ODEProblem):
    def setup(self):
        self.ode_system = Wrap(ODESystemModel)
        # parameters
        self.add_parameter('p1', dynamic=True, shape=(num, 1))
        self.add_parameter('p2', dynamic=True, shape=(num, 1))
        self.add_parameter('p3', dynamic=True, shape=(num, 1))
        # states
        self.add_state('u', 'du', initial_condition_name='u_0', output='u')
        self.add_state('w', 'dw', initial_condition_name='w_0', output='w')
        self.add_state('theta', 'dtheta', initial_condition_name='theta_0', output='theta')
        self.add_state('q', 'dq', initial_condition_name='q_0', output='q')
        self.add_state('x', 'dx', initial_condition_name='x_0', output='x')
        self.add_state('z', 'dz', initial_condition_name='z_0', output='z')
        self.add_state('e_lift', 'de_lift', initial_condition_name='e_lift_0', output='e_lift')
        self.add_state('e_cruise', 'de_cruise', initial_condition_name='e_cruise_0', output='e_cruise')
        # timestep vector
        self.add_times(step_vector='h')



# CSDL model specification containing the ODE integrator
class RunModel(csdl.Model):
    def define(self):
        dt = 0.5
        self.create_input('dt', dt)
        self.add(timestep(num=num))
        self.create_input('coefficients', np.ones(num+1)/(num+1))
        # create cubic spline control point vectors
        #p1c = np.array([900, 900, 800, 600, 0, 0, 0, 0, 0, 0])/60
        p1c = np.ones(4)*720/60
        #p1c = np.array([12.62486166, 10.78185929,  9.6379006,   7.60908334,  6.66133132,  2.93060859, 0., 0.10978526])
        self.create_input('p1c', p1c)
        #p2c = np.array([900, 900, 800, 600, 0, 0, 0, 0, 0, 0])/60
        p2c = np.ones(4)*720/60
        #p2c = np.array([12.62486166, 10.78185929,  9.6379006,   7.60908334,  6.66133132,  2.93060859, 0., 0.10978526])
        self.create_input('p2c', p2c)
        #p3c = np.array([0, 0, 150, 1500, 1500, 1500, 1500, 1500, 1500, 1500])/60 # rpm
        p3c = np.ones(4)*1100/60
        #p3c = np.array([21.04933512, 20.84450483, 20.74110146, 20.74749078, 20.85190087, 18.76607568, 10.61778928, 11.85133296])
        self.create_input('p3c', p3c)
        
        # create universal A matrix for spline calculations
        A, A_inv = A_matrix(num,p1c)
        self.create_input('A_inv', A_inv)
        # solve system and interpolate cubic splines
        
        self.add(p1_B(N=len(p1c)))
        self.add(p2_B(N=len(p2c)))
        self.add(p3_B(N=len(p3c)))
        self.add(p1_cubic(N=len(p1c)))
        self.add(p2_cubic(N=len(p2c)))
        self.add(p3_cubic(N=len(p3c)))
        self.add(p1_interpolate(num=num, N=len(p1c)))
        self.add(p2_interpolate(num=num, N=len(p2c)))
        self.add(p3_interpolate(num=num, N=len(p3c)))

        # initial conditions for states
        self.create_input('u_0', 0.1)
        self.create_input('w_0', 0)
        self.create_input('theta_0', 0)
        self.create_input('q_0', 0)
        self.create_input('x_0', 0)
        self.create_input('z_0', 100)
        self.create_input('e_lift_0', 0)
        self.create_input('e_cruise_0', 0)
        
        # create ODE model containing the integrator
        self.add(ODEProblem.create_solver_model(), 'subgroup', ['*'])
        # declare variables from integrator
        u = self.declare_variable('u', shape=(num+1,))
        w = self.declare_variable('w', shape=(num+1,))
        theta = self.declare_variable('theta', shape=(num+1,))
        q = self.declare_variable('q', shape=(num+1,))
        z = self.declare_variable('z', shape=(num+1,))
        e_lift = self.declare_variable('e_lift', shape=(num+1,))
        e_cruise = self.declare_variable('e_cruise', shape=(num+1,))
        p1 = self.declare_variable('p1', shape=(num,))
        p2 = self.declare_variable('p2', shape=(num,))
        """
        # final vertical velocity
        dz = u*csdl.sin(theta) - w*csdl.cos(theta)
        final_dz = dz[-1]
        self.register_output('final_dz', final_dz)
        self.add_constraint('final_dz', equals=0)
        """
        # altitude constraints
        final_z = z[-1]
        self.register_output('final_z', final_z)
        #self.add_constraint('final_z', equals=100)
        self.add_constraint('z', lower=99.7, upper=100.3)
        # final horizontal velocity
        dx = u*csdl.cos(theta) + w*csdl.sin(theta)
        final_dx = dx[-1]
        self.register_output('final_dx', final_dx)
        self.add_constraint('final_dx', lower=45)
        # final lift rpm
        final_p1 = p1[-1]
        self.register_output('final_p1', final_p1)
        self.add_constraint('final_p1', equals=0)

        """
        final_theta = theta[-1]
        self.register_output('final_theta', final_theta)
        self.add_constraint('final_theta', equals=0)
        """
        self.add_constraint('theta', lower=-0.1, upper=0.1)
        
        final_q = q[-1]
        self.register_output('final_q', final_q)
        self.add_constraint('final_q', equals=0)
        
        """
        rpm_ratio = p1/p2
        self.register_output('rpm_ratio', rpm_ratio)
        self.add_constraint('rpm_ratio', lower=0.999, upper=1.001)
        """
        # maximum power available constraint
        self.add(Power(num=num))
        self.add_constraint('lift_power', upper=829218)
        self.add_constraint('cruise_power', upper=468299)
        # calculate efficiency parameter
        self.add(Transition_Efficiency(num=num, m=3724))
        
        # create objective
        total_lift_energy = e_lift[-1]
        total_cruise_energy = e_cruise[-1]
        total_energy = total_cruise_energy + total_lift_energy*10
        self.register_output('objective', total_energy**0.5)
        """
        p1_integral = self.declare_variable('p1_integral', shape=(len(p1c)-1,1))
        p2_integral = self.declare_variable('p2_integral', shape=(len(p2c)-1,1))
        p3_integral = self.declare_variable('p3_integral', shape=(len(p3c)-1,1))
        objective = csdl.sum(p1_integral) + csdl.sum(p2_integral) + csdl.sum(p3_integral)
        self.register_output('objective', objective)
        """
        # add design variables
        #self.add_design_variable('dt', lower=0.2, upper=0.5) 
        self.add_design_variable('p1c', lower=-1)
        self.add_design_variable('p2c', lower=-1)
        self.add_design_variable('p3c', lower=-1)
        
        # add objective
        self.add_objective('objective')
        

t1 = time.perf_counter()

# ODEProblem_instance
num = 30 # multiple of number of control points
approach = 'time-marching'
ODEProblem = ODEProblemTest('ExplicitMidpoint', approach, num_times=num, display='default', visualization='end')

# Simulator Object:
sim = csdl_om.Simulator(RunModel(), mode='rev')

"""
# Setup your optimization problem (modopt SNOPT)
prob = CSDLProblem(
    problem_name='traj_optn',
    simulator=sim,)
# Setup your optimizer with the problem
optimizer = SNOPT(prob, 
                  Infinite_bound=1.0e20, 
                  Verify_level=3,
                  Major_iterations = 200, 
                  Major_optimality=1e-1, 
                  Major_feasibility=1e-1,)
# Check first derivatives at the initial guess, if needed
optimizer.check_first_derivatives(prob.x0)
# Solve your optimization problem
optimizer.solve()
"""
# csdl_om SLSQP
sim.prob.driver = om.ScipyOptimizeDriver()
sim.prob.driver.options['optimizer'] = 'SLSQP'
sim.prob.driver.options['tol'] = 1.0e-0
sim.prob.driver.options['maxiter'] = 100
sim.prob.run_driver()
#sim.run()

#print(sim['p1c'])
print('Transition Efficiency: ', sim['e'])
print('Total Energy: ', sim['objective']*1000000)

t2 = time.perf_counter()
delta_t = (t2 - t1)/60
print('elapsed time: ', delta_t)

# plotting
"""
plt.plot(sim['p1'],label='front lift rps')
plt.plot(sim['p2'],label='rear lift rps')
plt.plot(sim['p3'],label='cruise rps')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('rps')
plt.title('rps')
"""
"""
plt.plot(sim['cruise_power'])
plt.plot(sim['lift_power'])
"""
"""
plt.plot(sim['x'],sim['z'])
plt.xlabel('horizontal distance (m)')
plt.ylabel('vertical distance (m)')
plt.title('trajectory')


plt.plot(sim['theta'],label='theta')
plt.plot(sim['q'],label='q')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('theta (rad), q (rad/s)')
plt.title('theta & q')

plt.plot(sim['u'],label='u')
plt.plot(-1*sim['w'],label='w')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('u, w (m/s)')
plt.title('u & w')
"""




