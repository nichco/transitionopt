from ozone.api import ODEProblem, Wrap, NativeSystem
import csdl
import numpy as np

import sys
sys.path.append("C:/Users/Nicholas Orndorff/Desktop/LSDO/Aviation Code 2022/bemv2")
from runbemv2 import BEM
#from momentum import Glauert
from density import Density


class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes')

    def define(self):
        # Required every time for ODE systems or Profile Output systems
        n = self.parameters['num_nodes']
        # states
        u = self.create_input('u', shape=n)
        w = self.create_input('w', shape=n)
        theta = self.create_input('theta', shape=n)
        q = self.create_input('q', shape=n)
        x = self.create_input('x', shape=n)
        z = self.create_input('z', shape=n)
        e_lift = self.create_input('e_lift', shape=n)
        e_cruise = self.create_input('e_cruise', shape=n)
        # inputs
        p1 = self.create_input('p1', shape=(n))
        p2 = self.create_input('p2', shape=(n))
        p3 = self.create_input('p3', shape=(n))
        
        # constants
        g = 9.81 # acceleration due to gravity m/s^2
        #rho = 1.225 # density kg/m^3
        self.add(Density(out_name='rho_m', alt_name='z'))
        rho = self.declare_variable('rho_m')
        m = 3724 # mass kg
        Iyy = 1000 # moment of inertia
        cla = 2*np.pi # lift slope cl/rad
        aefs = 0.1 # wing angle of incidence rad
        e = 0.8 # Oswald's efficiency factor
        cd0 = 0.0151 # zero lift drag coefficient
        AR = 11.2 # aspect ratio
        s = 39 # 39 18.1 wing area m^2
        
        
        # forces and moments
        alpha = csdl.arctan(w/u)
        v = (u**2 + w**2)**0.5
        cl = (alpha + aefs)*cla
        cd = cd0 + (cl**2)/(np.pi*e*AR)
        lift = 0.5*rho*(v**2)*s*cl
        drag = 0.5*rho*(v**2)*s*cd
            
        fax = -drag*csdl.cos(alpha) + lift*csdl.sin(alpha)
        faz = -drag*csdl.sin(alpha) - lift*csdl.cos(alpha)
        
        
        w_inv = -1*w
        self.register_output('w_inv', w_inv)
        self.create_input('Vy', val=1)
        self.create_input('Vz', val=1)
        
        self.add(BEM(r_name='p1',  
                     vx_name='w_inv', 
                     vy_name='Vy', 
                     vz_name='Vz',
                     tag='_1',
                     rotor_diameter=3.2,
                     num_blades=2))
        t1 = self.declare_variable('total_thrust_1')
        q1 = self.declare_variable('total_torque_1')
        power_1 = 2*np.pi*q1*p1
        
        self.add(BEM(r_name='p2',  
                     vx_name='w_inv', 
                     vy_name='Vy', 
                     vz_name='Vz',
                     tag='_2',
                     rotor_diameter=3.2,
                     num_blades=2))
        t2 = self.declare_variable('total_thrust_2')
        q2 = self.declare_variable('total_torque_2')
        power_2 = 2*np.pi*q2*p2
        
        self.add(BEM(r_name='p3',  
                     vx_name='u', 
                     vy_name='Vy', 
                     vz_name='Vz',
                     tag='_3',
                     rotor_diameter=2.85,# 2.85
                     num_blades=6))
        t3 = self.declare_variable('total_thrust_3')
        q3 = self.declare_variable('total_torque_3')
        power_3 = 2*np.pi*q3*p3
        

        fpx = 1*t3
        fpz = -1*(4*t1 + 4*t2)
        mp = 4*t1 - 4*t2

        #fpx = 0
        #fpz = -(8*t1)
        #mp = 0*u
        
        #fpx = 1*p3
        #fpz = -1*(p1 + p2)
        ma = 0
        #mp = p1 - p2
        
        # system of ODE's
        du = -q*w - g*csdl.sin(theta) + (fax + fpx)/m
        dw = q*u + g*csdl.cos(theta) + (faz + fpz)/m
        dtheta = 1*q
        dq = (ma + mp)/Iyy
        dx = u*csdl.cos(theta) + w*csdl.sin(theta)
        dz = u*csdl.sin(theta) - w*csdl.cos(theta)
        de_lift = (4*power_1 + 4*power_2)/1000000
        de_cruise = power_3/1000000
        #de_lift = (8*power_1)/1000000
        
        # register outputs
        self.register_output('du', du)
        self.register_output('dw', dw)
        self.register_output('dtheta', dtheta)
        self.register_output('dq', dq)
        self.register_output('dx', dx)
        self.register_output('dz', dz)
        self.register_output('de_lift', de_lift)
        self.register_output('de_cruise', de_cruise)
        
        
        
        
        
        
        