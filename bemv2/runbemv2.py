import numpy as np 
import csdl
import csdl_om

from BEM_group import BEMGroup

from get_surrogate_model import get_surrogate_model
from get_rotor_dictionary import get_rotor_dictionary
from get_max_LD_parameters_reference_chord import get_max_LD_parameters_reference_chord

import time


class Velocity_Conversion(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_tangential', types=int)
        self.parameters.declare('vx_name', types=str, default='Vx')
        self.parameters.declare('vy_name', types=str, default='Vy')
        self.parameters.declare('vz_name', types=str, default='Vz')
        self.parameters.declare('tag', types=str)
    def define(self):
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        vx_name = self.parameters['vx_name']
        vy_name = self.parameters['vy_name']
        vz_name = self.parameters['vz_name']
        tag = self.parameters['tag']
        shape = (num_evaluations, num_radial, num_tangential)
        
        Vx = self.declare_variable(vx_name)
        Vy = self.declare_variable(vy_name)
        Vz = self.declare_variable(vz_name)
        
        inflow_velocity = self.create_output('inflow_velocity'+tag, shape=shape + (3,))
        
        for i in range(num_evaluations):
                for j in range(num_radial):    
                    for k in range(num_tangential):    
                        inflow_velocity[i, j, k, 0] = csdl.expand(Vx, (1,1,1,1)) 
                        inflow_velocity[i, j, k, 1] = csdl.expand(Vy, (1,1,1,1)) 
                        inflow_velocity[i, j, k, 2] = csdl.expand(Vz, (1,1,1,1))  
        

class BEM(csdl.Model):
    def initialize(self):
        self.parameters.declare('vx_name', types=str, default='Vx')
        self.parameters.declare('vy_name', types=str, default='Vy')
        self.parameters.declare('vz_name', types=str, default='Vz')
        self.parameters.declare('r_name', types=str, default='rotational_speed')
        self.parameters.declare('tag', types=str, default='')
        self.parameters.declare('rotor_diameter')
        self.parameters.declare('num_blades')
    def define(self):
        vx_name = self.parameters['vx_name']
        vy_name = self.parameters['vy_name']
        vz_name = self.parameters['vz_name']
        r_name = self.parameters['r_name']
        tag = self.parameters['tag']
        rotor_diameter = self.parameters['rotor_diameter']
        num_blades = self.parameters['num_blades']
        
        mode = 2
        # The following airfoils are currently available: 'NACA_4412', 'Clark_Y', 'NACA_0012', 'mh117'; We recommend 'NACA_4412' or 'Clark_Y'
        airfoil             = 'Clark_Y' 
        interp              = get_surrogate_model(airfoil)
        RPM                 = 1500
        # x, y, z velocity components (m/s); Vx is the axial velocity component 
        Vx                  = 10   # Axial inflow velocity (i.e. V_inf)
        # Specify number of blades and altitude
        altitude            = 100        # (in m)
        # specify three parameters for optimal blade design 
        reference_radius    = rotor_diameter / 2  # Specify the reference radius; We recommend radius / 2
        reference_chord     = 0.25                # Specify the reference chord length at the reference radius (in m)
        # The following parameters are used for mode 2 only
        # Change these parameters if you want to chord and twist profile to vary linearly from rotor hub to tip
        root_chord          = 0.3       # Chord length at the root/hub
        root_twist          = 80        # Twist angle at the blade root/hub (deg)
        tip_chord           = 0.2       # Chord length at the tip
        tip_twist           = 20        # Twist angle at the blade tip (deg)
        # The following parameters specify the radial and tangential mesh as well as the number of time steps; 
        num_evaluations     = 1         # Discretization in time:                 Only required if your operating conditions change in time
        num_radial          = 30        # Discretization in spanwise direction:   Should always be at least 25
        num_tangential      = 1         # Discretization in tangential direction: Only required if Vy,Vz are non-zero; recommend at least 20
        #---- ---- ---- ---- ---- ---- ---- ---- ---- END OF USER SPECIFIED INPUT ---- ---- ---- ---- ---- ---- ---- ---- ---- #
        ideal_alpha_ref_chord, Cl_max, Cd_min = get_max_LD_parameters_reference_chord(interp, reference_chord, reference_radius, Vx, RPM, altitude)
        rotor = get_rotor_dictionary(airfoil, num_blades, altitude, mode, interp, ideal_alpha_ref_chord, Cl_max, Cd_min,reference_chord)
        
        
        
        rotor_model = csdl.Model()
        
        group = BEMGroup(
            mode = mode,
            rotor=rotor,
            num_evaluations=num_evaluations,
            num_radial=num_radial,
            num_tangential=num_tangential,
            r_name=r_name,
            tag=tag,
        )
        rotor_model.add(group,'BEM_group')#, promotes = ['*'])
        
        self.create_input('reference_radius'+tag, reference_radius)
        #rotational_speed = RPM/60
        rotational_speed = self.declare_variable('rotational_speed')
        #self.create_input('rotational_speed', rotational_speed)
        #self.create_input('reference_rotational_speed', rotational_speed)
        self.create_input('reference_axial_inflow_velocity'+tag, 50)
        rotor_radius = rotor_diameter / 2
        self.create_input('rotor_radius'+tag, rotor_radius)
        hub_radius = 0.2*rotor_radius
        self.create_input('hub_radius'+tag, hub_radius)
        slice_thickness = (rotor_radius - hub_radius)/(num_radial - 1)
        self.create_input('slice_thickness'+tag, slice_thickness)
        chord = np.linspace(
                root_chord,
                tip_chord,
                num_radial,
            )
        self.create_input('chord'+tag, chord)
        pitch = np.linspace(
                root_twist * np.pi / 180.,
                tip_twist * np.pi / 180.,
                num_radial,
            )
        self.create_input('pitch'+tag, pitch)
        self.create_input('reference_chord'+tag, reference_chord)
        reference_blade_solidity = num_blades * reference_chord / 2. / np.pi / reference_radius
        self.create_input('reference_blade_solidity'+tag, reference_blade_solidity)
        reference_tangential_inflow_velocity = rotational_speed * 2. * np.pi * reference_radius
        self.register_output('reference_tangential_inflow_velocity'+tag, reference_tangential_inflow_velocity)
        #self.create_input('reference_tangential_inflow_velocity', reference_tangential_inflow_velocity)
        x_dir = np.zeros((num_evaluations,3))
        y_dir = np.zeros((num_evaluations,3))
        z_dir = np.zeros((num_evaluations,3))
        for i in range(num_evaluations):
                x_dir[i, :] = [1., 0., 0.]
                y_dir[i, :] = [0., 1., 0.]
                z_dir[i, :] = [0., 0., 1.]
        self.create_input('x_dir'+tag, x_dir)
        self.create_input('y_dir'+tag, y_dir)
        self.create_input('z_dir'+tag, z_dir)
        
        self.add(Velocity_Conversion(num_evaluations=num_evaluations, 
                                     num_radial=num_radial, 
                                     num_tangential=num_tangential,
                                     vx_name=vx_name,
                                     vy_name=vy_name,
                                     vz_name=vz_name,
                                     tag=tag,))
        
        self.add(rotor_model)


"""
class Run(csdl.Model):
    def define(self):
        self.create_input('u', val=10)
        self.create_input('Vy', val=0)
        self.create_input('w', val=0)
        
        RPM = 100
        self.create_input('p3', RPM/60)
        
        self.add(BEM(r_name='p3', 
                     vx_name='u', 
                     vy_name='Vy', 
                     vz_name='w',
                     tag='_1',
                     rotor_diameter=2,
                     num_blades=3))
        t1 = self.declare_variable('total_thrust_1')
        q1 = self.declare_variable('total_torque_1')
        power_1 = 2*np.pi*q1*p1

t1 = time.perf_counter()
sim = csdl_om.Simulator(Run())
sim.run()
print(sim['total_thrust_1'])

t2 = time.perf_counter()
delta_t = (t2 - t1)/60
print('elapsed time: ', delta_t)

"""

