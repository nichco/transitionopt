import numpy as np
from csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import csdl

#from airfoil_surrogate_model_group import AirfoilSurrogateModelGroup
#from atmosphere_group import AtmosphereGroup

class PhiBracketedSearchGroup(Model):

    def initialize(self):
        self.parameters.declare('rotor')
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('mode', types = int)
        self.parameters.declare('tag', types=str)

    def define(self):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']
        mode = self.parameters['mode']
        tag = self.parameters['tag']
        B = num_blades = rotor['num_blades']
        
        model = Model()
        
        sigma = model.declare_variable('_blade_solidity'+tag, shape=shape)
        Vx = model.declare_variable('_axial_inflow_velocity'+tag, shape=shape)
        Vt = model.declare_variable('_tangential_inflow_velocity'+tag, shape=shape)
        radius = model.declare_variable('_radius'+tag,shape= shape)
        rotor_radius = model.declare_variable('_rotor_radius'+tag, shape= shape)
        hub_radius = model.declare_variable('_hub_radius'+tag, shape = shape)
        chord = model.declare_variable('chord_distribution'+tag, shape=shape)
        twist = model.declare_variable('pitch_distribution'+tag, shape=shape)
        
        # phi is state (inflow angle) we're solving for in the bracketed search
        phi = model.declare_variable('phi_distribution'+tag, shape = shape)
        """
        # Adding atmosphere group to compute Reynolds number 
        model.add(AtmosphereGroup(
            shape = shape,
            rotor = rotor,
            mode = mode,
            tag = tag
        ), name = 'atmosphere_group', promotes = ['*'])
        """
        Re = model.declare_variable('Re'+tag, shape = shape)
        
        alpha = twist - phi
        model.register_output('alpha_distribution'+tag, alpha)
        """
        # Adding custom component to embed airfoil model in the bracketed search
        airfoil_model_output = csdl.custom(Re,alpha,chord, op= AirfoilSurrogateModelGroup(
            rotor = rotor,
            shape = shape,
            tag = tag
        ))
        model.register_output('Cl'+tag, airfoil_model_output[0])
        model.register_output('Cd'+tag, airfoil_model_output[1])
        
        Cl = airfoil_model_output[0]
        Cd = airfoil_model_output[1]
        """
        # simple replacement for surrogate model
        Cl0 = 0
        Cl1 = 2*np.pi
        Cd0 = 0.005
        Cd1 = 0
        Cd2 = 0.5
        
        Cl = Cl0 + Cl1 * alpha
        Cd = Cd0 + Cd1 * alpha + Cd2 * alpha**2
        model.register_output('Cl'+tag, Cl)
        model.register_output('Cd'+tag, Cd)
        
        # end of modified code
        
        
        # Embedding Prandtl tip losses 
        f_tip = B / 2 * (rotor_radius - radius) / radius / csdl.sin(phi)
        f_hub = B / 2 * (radius - hub_radius) / hub_radius / csdl.sin(phi)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

        F = F_tip * F_hub
        model.register_output('F'+tag,F)

        # Setting up residual function
        Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
        Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

        model.register_output('Cx'+tag,Cx)
        model.register_output('Ct'+tag,Ct)

        term1 = Vt * (sigma * Cx - 4 * F * csdl.sin(phi)**2)
        term2 = Vx * (2 * F * csdl.sin(2 * phi) + Ct * sigma)
        
        BEM_residual = term1 + term2
        
        model.register_output('BEM_residual_function'+tag, BEM_residual)
        
        # Solving residual function for state phi 
        eps = 1e-7
        # print(eps)
        solve_BEM_residual = self.create_implicit_operation(model)
        solve_BEM_residual.declare_state('phi_distribution'+tag, residual='BEM_residual_function'+tag, bracket=(eps, np.pi/2 - eps))

        # sigma = model.declare_variable('_blade_solidity', shape=shape)
        # Vx = model.declare_variable('_axial_inflow_velocity', shape=shape)
        # Vt = model.declare_variable('_tangential_inflow_velocity', shape=shape)
        # radius = model.declare_variable('_radius',shape= shape)
        # rotor_radius = model.declare_variable('_rotor_radius', shape= shape)
        # hub_radius = model.declare_variable('_hub_radius', shape = shape)
        # chord = model.declare_variable('chord_distribution',shape=shape)
        # twist = model.declare_variable('pitch_distribution', shape=shape)
        # Re = model.declare_variable('Re', shape = shape)
    

        phi, Cl, Cd,F, Cx, Ct = solve_BEM_residual(sigma,Vx,Vt,radius,rotor_radius,hub_radius,chord,twist,Re, expose = ['Cl'+tag, 'Cd'+tag,'F'+tag,'Cx'+tag,'Ct'+tag])

            


