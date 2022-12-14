import numpy as np
from csdl import Model
import csdl

from rotor_parameters import RotorParameters

class AtmosphereGroup(Model):

    def initialize(self):
        self.parameters.declare('shape', types = tuple)
        self.parameters.declare('mode', types = int)
        self.parameters.declare('rotor')
        self.parameters.declare('tag', types=str)

    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']
        mode  = self.parameters['mode']
        tag = self.parameters['tag']

        altitude    = rotor['altitude'] * 1e-3

        # Constants
        L           = 6.5
        R           = 287
        T0          = 288.16
        P0          = 101325
        g0          = 9.81
        mu0         = 1.735e-5
        S1          = 110.4


        chord       = self.declare_variable('chord_distribution'+tag, shape=shape)
        Vx          = self.declare_variable('_axial_inflow_velocity'+tag, shape=shape)
        Vt          = self.declare_variable('_tangential_inflow_velocity'+tag, shape=shape)

        # Temperature 
        T           = T0 - L * altitude

        # Pressure 
        P           = P0 * (T/T0)**(g0/(L * 1e-3)/R)
        
        # Density
        rho         = P/R/T * Vt / Vt
        
        # Dynamic viscosity (using Sutherland's law)  
        mu          = mu0 * (T/T0)**(3/2) * (T0 + S1)/(T + S1)

        # Reynolds number
        W           = (Vx**2 + Vt**2)**0.5
        Re          = rho * W * chord / mu


        self.register_output('Re'+tag, Re)
        self.register_output('rho'+tag, rho)
            

        
