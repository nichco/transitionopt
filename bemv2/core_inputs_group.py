import numpy as np
from csdl import Model
import csdl


class CoreInputsGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_tangential', types=int)
        self.parameters.declare('r_name', types=str)
        self.parameters.declare('tag', types=str)

    def define(self):
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        r_name = self.parameters['r_name']
        tag = self.parameters['tag']
        shape = (num_evaluations, num_radial, num_tangential)

        hub_radius = self.declare_variable('hub_radius'+tag)
        rotor_radius = self.declare_variable('rotor_radius'+tag)
        slice_thickness = self.declare_variable('slice_thickness'+tag)
        reference_chord = self.declare_variable('reference_chord'+tag)
        reference_radius = self.declare_variable('reference_radius'+tag)
        alpha = self.declare_variable('alpha'+tag)
        alpha_stall = self.declare_variable('alpha_stall'+tag)
        alpha_stall_minus = self.declare_variable('alpha_stall_minus'+tag)
        AR = self.declare_variable('AR'+tag)

        position = self.declare_variable('position'+tag, shape=(num_evaluations,3))
        x_dir = self.declare_variable('x_dir'+tag, shape=(num_evaluations, 3))
        y_dir = self.declare_variable('y_dir'+tag, shape=(num_evaluations, 3))
        z_dir = self.declare_variable('z_dir'+tag, shape=(num_evaluations, 3))
        inflow_velocity = self.declare_variable('inflow_velocity'+tag, shape=shape + (3,))
        pitch = self.declare_variable('pitch'+tag, shape=(num_radial,))
        chord = self.declare_variable('chord'+tag, shape=(num_radial,))
        
        rotational_speed = self.declare_variable(r_name)
        direction = self.create_input('direction'+tag, val=1., shape=num_evaluations)

        self.register_output('_hub_radius'+tag, csdl.expand(hub_radius, shape))
        self.register_output('_rotor_radius'+tag, csdl.expand(rotor_radius, shape))
        self.register_output('_slice_thickness'+tag, csdl.expand(slice_thickness, shape))
        self.register_output('_reference_chord'+tag, csdl.expand(reference_chord,shape))
        self.register_output('_reference_radius'+tag, csdl.expand(reference_radius,shape))

        self.register_output('_position'+tag, csdl.expand(position, shape + (3,), 'il->ijkl'))
        self.register_output('_x_dir'+tag, csdl.expand(x_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_y_dir'+tag, csdl.expand(y_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_z_dir'+tag, csdl.expand(z_dir, shape + (3,), 'il->ijkl'))
        self.register_output('_inflow_velocity'+tag, 1. * inflow_velocity)
        self.register_output('pitch_distribution'+tag, csdl.expand(pitch, shape, 'j->ijk'))
        self.register_output('chord_distribution'+tag, csdl.expand(chord, shape, 'j->ijk'))

        self.register_output('_rotational_speed'+tag, csdl.expand(rotational_speed, shape))

        _theta = np.einsum(
            'ij,k->ijk',
            np.ones((num_evaluations, num_radial)),
            np.linspace(0., 2. * np.pi, num_tangential),
        )
        self.create_input('_theta'+tag, val=_theta)

        normalized_radial_discretization = 1. / num_radial / 2. \
            + np.linspace(0., 1. - 1. / num_radial, num_radial)

        _normalized_radius = np.einsum(
            'ik,j->ijk',
            np.ones((num_evaluations, num_tangential)),
            normalized_radial_discretization,
        )
        self.create_input('_normalized_radius'+tag, val=_normalized_radius)