import numpy as np
from csdl import Model
import csdl


class PreprocessGroup(Model):

    def initialize(self):
        self.parameters.declare('rotor')#, types=RotorParameters)
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('tag', types=str)

    def define(self):
        rotor = self.parameters['rotor']
        shape = self.parameters['shape']
        tag = self.parameters['tag']

        num_blades = rotor['num_blades']

        # -----
        _rotational_speed = self.declare_variable('_rotational_speed'+tag, shape=shape)
        chord_distribution = self.declare_variable('chord_distribution'+tag, shape=shape)
        _hub_radius = self.declare_variable('_hub_radius'+tag, shape=shape)
        _rotor_radius = self.declare_variable('_rotor_radius'+tag, shape=shape)
        _theta = self.declare_variable('_theta'+tag, shape=shape)
        _normalized_radius = self.declare_variable('_normalized_radius'+tag, shape=shape)

        _inflow_velocity = self.declare_variable('_inflow_velocity'+tag, shape=shape + (3,))
        _x_dir = self.declare_variable('_x_dir'+tag, shape=shape + (3,))
        _y_dir = self.declare_variable('_y_dir'+tag, shape=shape + (3,))
        _z_dir = self.declare_variable('_z_dir'+tag, shape=shape + (3,))
        _direction = self.declare_variable('_direction'+tag, shape=shape)

        _reference_chord = self.declare_variable('_reference_chord'+tag, shape=shape)
        _reference_radius = self.declare_variable('_reference_radius'+tag, shape=shape)

        # -----
        _angular_speed = 2 * np.pi * _rotational_speed
        self.register_output('_angular_speed'+tag, _angular_speed)

        _radius = _hub_radius + (_rotor_radius - _hub_radius) * _normalized_radius
        self.register_output('_radius'+tag, _radius)

        _ref_radius = _hub_radius + (_rotor_radius - _hub_radius) * _normalized_radius
        self.register_output('_ref_radius'+tag,_ref_radius)
        # -----

        self.register_output('_blade_solidity'+tag, num_blades * chord_distribution / 2. / np.pi / _radius)
        self.register_output('_reference_blade_solidity'+tag, num_blades * _reference_chord / 2. / np.pi / _reference_radius)
        # -----

        
        _inflow_x = csdl.einsum(_inflow_velocity, _x_dir, subscripts='ijkl,ijkl->ijk')
        _inflow_y = csdl.einsum(_inflow_velocity, _y_dir, subscripts='ijkl,ijkl->ijk')
        _inflow_z = csdl.einsum(_inflow_velocity, _z_dir, subscripts='ijkl,ijkl->ijk')
        
        self.register_output('_axial_inflow_velocity'+tag, _inflow_x)

        self.register_output('inflow_y'+tag,_inflow_y)
        self.register_output('inflow_z'+tag, _inflow_z)

        self.register_output(
            '_tangential_inflow_velocity'+tag, 
            _direction * _inflow_y * csdl.sin(_theta) - 
            _direction * _inflow_z * csdl.cos(_theta) + 
            _radius * _angular_speed
        )

