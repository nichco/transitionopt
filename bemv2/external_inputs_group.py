import numpy as np
from csdl import Model
import csdl


class ExternalInputsGroup(Model):
    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_tangential', types=int)
        self.parameters.declare('tag', types=str)

    def define(self):
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        tag = self.parameters['tag']
        shape = (num_evaluations, num_radial, num_tangential)


        #self.create_input('reference_radius', shape=1)
        self.create_input('reference_position'+tag, shape=(1,3))
        self.create_input('reference_pitch'+tag, shape=1)
        #self.create_input('reference_chord', shape=1)
        self.create_input('reference_twist'+tag, shape=1)
        #self.create_input('reference_axial_inflow_velocity', shape=1)
        #self.create_input('reference_blade_solidity', shape =1)
        #self.create_input('reference_tangential_inflow_velocity', shape=1)
        #self.create_input('reference_rotational_speed',shape=1)

        #self.create_input('hub_radius', shape=1)
        #self.create_input('rotor_radius', shape=1)

        #self.create_input('slice_thickness', shape =1)
        self.create_input('position'+tag, shape=(num_evaluations, 3))
        #self.create_input('x_dir', shape=(num_evaluations, 3))
        #self.create_input('y_dir', shape=(num_evaluations, 3))
        #self.create_input('z_dir', shape=(num_evaluations, 3))
        #self.create_input('inflow_velocity', shape=shape + (3,))
        #self.create_input('rotational_speed', shape = 1)

        #self.create_input('pitch', shape = (num_radial,))
        #self.create_input('chord', shape = (num_radial,))

        # self.create_input('pitch_cp', shape = (pitch_num_cp,))
        # self.create_input('chord_cp', shape = (chord_num_cp,))
