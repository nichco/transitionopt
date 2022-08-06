import numpy as np
from csdl import Model
import csdl

class InducedVelocityGroup(Model):
    
    def initialize(self):
        self.parameters.declare('rotor')
        self.parameters.declare('mode', types = int)
        self.parameters.declare('shape',types = tuple)
        self.parameters.declare('tag', types=str)

    def define(self):
        #rotor = self.parameters['rotor']
        #mode = self.parameters['mode']
        shape = self.parameters['shape']
        tag = self.parameters['tag']

        #B = num_blades = rotor['num_blades']

        #print('TEST')
        phi = self.declare_variable('phi_distribution'+tag, shape=shape)
        #twist = self.declare_variable('pitch_distribution'+tag, shape=shape)

        Vx = self.declare_variable('_axial_inflow_velocity'+tag, shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity'+tag, shape=shape)
        #Vx_ref = self.declare_variable('reference_axial_inflow_velocity'+tag)
        
        sigma = self.declare_variable('_blade_solidity'+tag, shape=shape)
        #chord = self.declare_variable('chord_distribution'+tag,shape=shape)
        radius = self.declare_variable('_radius'+tag, shape = shape)
        dr = self.declare_variable('_slice_thickness'+tag, shape = shape)
        rho = self.declare_variable('rho'+tag, shape = shape)
        
        F = self.declare_variable('F'+tag, shape=shape)
        """
        rotational_speed = self.declare_variable('_rotational_speed'+tag, shape=shape)
        n = self.declare_variable('reference_rotational_speed'+tag)
        """
        Cl = self.declare_variable('Cl'+tag, shape=shape)
        Cd = self.declare_variable('Cd'+tag, shape=shape)

        #Cx1 = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
        Ct1 = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

        #Cx2 = self.declare_variable('Cx'+tag, shape=shape)
        Ct2 = self.declare_variable('Ct'+tag, shape=shape)

        ux = (4 * F * Vt * csdl.sin(phi)**2) / (4 * F * csdl.sin(phi) * csdl.cos(phi) +  sigma * Ct2)
        #ux_2 = Vx + sigma * Cx2 * Vt / (4 * F * csdl.sin(phi) * csdl.cos(phi) + sigma * Ct1)
    
        ut = 2 * Vt * sigma * Ct2 / (2 * F * csdl.sin(2 * phi) + sigma * Ct1)

        dT = 4 * np.pi * radius * rho * ux * (ux - Vx) * F * dr
        dQ = 2 * np.pi * radius**2 * rho * ux * ut * F * dr
        """
        dT2 = num_blades * Cx1 * 0.5 * rho * (ux**2 + (Vt - 0.5 * ut)**2) * chord * dr
        dQ2 = num_blades * Ct1 * 0.5 * rho * (ux**2 + (Vt - 0.5 * ut)**2) * chord * dr * radius
        
        T2 = csdl.sum(dT2)
        Q2 = csdl.sum(dQ2)
        """
        T = csdl.sum(dT)
        Q = csdl.sum(dQ)
        """
        self.register_output('_ux',ux)
        self.register_output('_ux_2',ux_2)
        self.register_output('_ut', ut)
        """
        #self.register_output('_local_thrust', dT)
        
        #self.register_output('total_thrust', T)
        self.register_output('total_thrust'+tag, T)
        
        #self.register_output('_local_thrust_2', dT2)
        #self.register_output('total_thrust_2', T2)


        #self.register_output('_local_torque', dQ)
        self.register_output('total_torque'+tag, Q)
        #self.register_output('_local_torque_2', dQ2)
        #self.register_output('total_torque_2', Q2)



