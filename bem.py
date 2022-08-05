import sys
sys.path.append("C:/Users/Nicholas Orndorff/lsdo_rotor/lsdo_rotor")

import csdl
import csdl_om
import numpy as np

from lsdo_rotor.inputs.external_inputs_group import ExternalInputsGroup
from lsdo_rotor.core.BEM_group import BEMGroup
from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.functions.get_rotor_dictionary import get_rotor_dictionary
from lsdo_rotor.functions.get_max_LD_parameters_reference_chord import get_max_LD_parameters_reference_chord 

class Expand_Velocity(csdl.Model):
    def define(self):
        u = self.declare_variable('u')
        w = self.declare_variable('w')
        
        num_evaluations     = 1         # Discretization in time   
        num_radial          = 30        # Discretization in spanwise direction
        num_tangential      = 1
        
        inflow_velocity = self.create_output('inflow_velocity', shape=(num_evaluations, num_radial, num_tangential, 3))
        
        for i in range(num_evaluations):
            for j in range(num_radial):    
                for k in range(num_tangential):    
                    inflow_velocity[i, j, k, 0] = csdl.expand(u, (1,1,1,1))
                    inflow_velocity[i, j, k, 2] = csdl.expand(w, (1,1,1,1))
        


class BEM(csdl.Model):
    def initialize(self):
        pass
    def define(self):
        """
        mode = 2
        airfoil             = 'Clark_Y' 
        interp              = get_surrogate_model(airfoil)
        rotor_diameter      = 2        # (in m) modify here AND external_inputs_group.py
        RPM                 = 1500
        # x, y, z velocity components (m/s); Vx is the axial velocity component 
        Vx                  = 60   # Axial inflow velocity (i.e. V_inf) 
        Vy                  = 0   # Side slip velocity in the propeller plane
        Vz                  = 0   # Side slip velocity in the propeller plane
        # Specify number of blades and altitude
        num_blades          = 3
        altitude            = 100        # (in m)
        # The following 3 parameters are used for mode 1 only! The user has to specify three parameters for optimal blade design 
        reference_radius    = rotor_diameter / 4  # Specify the reference radius; We recommend radius / 2
        reference_chord     = 0.15                # Specify the reference chord length at the reference radius (in m)
        use_external_rotor_geometry = 'n'
        # The following parameters specify the radial and tangential mesh as well as the number of time steps; 
        num_evaluations     = 1         # Discretization in time:                 Only required if your operating conditions change in time   
        num_radial          = 30        # Discretization in spanwise direction:   Should always be at least 25
        num_tangential      = 1         # Discretization in tangential direction: Only required if Vy,Vz are non-zero; recommend at least 20
        #---- ---- ---- ---- ---- ---- ---- ---- ---- END OF USER SPECIFIED INPUT ---- ---- ---- ---- ---- ---- ---- ---- ---- #
        
        RPM=1500
        ideal_alpha_ref_chord, Cl_max, Cd_min = get_max_LD_parameters_reference_chord(interp, reference_chord, reference_radius, Vx, RPM, altitude)
        rotor = get_rotor_dictionary(airfoil, num_blades, altitude, mode, interp, ideal_alpha_ref_chord, Cl_max, Cd_min,reference_chord)
        """
        
        rotor_model = csdl.Model()
        
        group = BEMGroup(
            mode = mode,
            rotor=rotor,
            num_evaluations=num_evaluations,
            num_radial=num_radial,
            num_tangential=num_tangential,
        )
        rotor_model.add(group,'BEM_group')#, promotes = ['*'])
        
        self.create_input('rotational_speed', val=RPM/60)
        self.create_input('reference_rotational_speed', val=RPM/60)
        
        self.add(Expand_Velocity())
        self.add(rotor_model)
        
        
        
mode = 2
airfoil             = 'Clark_Y' 
interp              = get_surrogate_model(airfoil)
rotor_diameter      = 2        # (in m) modify here AND external_inputs_group.py
RPM                 = 1500
# x, y, z velocity components (m/s); Vx is the axial velocity component 
Vx                  = 60   # Axial inflow velocity (i.e. V_inf) 
Vy                  = 0   # Side slip velocity in the propeller plane
Vz                  = 0   # Side slip velocity in the propeller plane
# Specify number of blades and altitude
num_blades          = 3
altitude            = 100        # (in m)
# The following 3 parameters are used for mode 1 only! The user has to specify three parameters for optimal blade design 
reference_radius    = rotor_diameter / 4  # Specify the reference radius; We recommend radius / 2
reference_chord     = 0.15                # Specify the reference chord length at the reference radius (in m)
use_external_rotor_geometry = 'n'
# The following parameters specify the radial and tangential mesh as well as the number of time steps; 
num_evaluations     = 1         # Discretization in time:                 Only required if your operating conditions change in time   
num_radial          = 30        # Discretization in spanwise direction:   Should always be at least 25
num_tangential      = 1         # Discretization in tangential direction: Only required if Vy,Vz are non-zero; recommend at least 20
#---- ---- ---- ---- ---- ---- ---- ---- ---- END OF USER SPECIFIED INPUT ---- ---- ---- ---- ---- ---- ---- ---- ---- #
ideal_alpha_ref_chord, Cl_max, Cd_min = get_max_LD_parameters_reference_chord(interp, reference_chord, reference_radius, Vx, RPM, altitude)
rotor = get_rotor_dictionary(airfoil, num_blades, altitude, mode, interp, ideal_alpha_ref_chord, Cl_max, Cd_min,reference_chord)
        
"""
sim = csdl_om.Simulator(BEM())
sim.run()
print(sim['total_thrust'])
"""
        
        
        
        
        
        
        
        
        
        