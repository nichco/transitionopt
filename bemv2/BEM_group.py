import numpy as np
from csdl import Model
import csdl

from rotor_parameters import RotorParameters
from external_inputs_group import ExternalInputsGroup
from core_inputs_group import CoreInputsGroup
from preprocess_group import PreprocessGroup
from atmosphere_group import AtmosphereGroup
from airfoil_surrogate_model_group import AirfoilSurrogateModelGroup
from phi_bracketed_search_group import PhiBracketedSearchGroup
from induced_velocity_group import InducedVelocityGroup

class BEMGroup(Model):

    def initialize(self):
        self.parameters.declare('rotor')#, types=RotorParameters)
        self.parameters.declare('mode', types=int)
        self.parameters.declare('num_evaluations', types=int)
        self.parameters.declare('num_radial', types=int)
        self.parameters.declare('num_tangential', types=int)
        self.parameters.declare('r_name', types=str)
        self.parameters.declare('tag', types=str)

    def define(self):
        num_evaluations = self.parameters['num_evaluations']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        mode = self.parameters['mode']
        rotor = self.parameters['rotor']
        r_name = self.parameters['r_name']
        tag = self.parameters['tag']

        shape = (num_evaluations, num_radial, num_tangential)        

        self.add(ExternalInputsGroup(
            shape = shape,
            num_evaluations = num_evaluations,
            num_radial = num_radial,
            num_tangential = num_tangential,
            tag = tag,
        ), name = 'external_inputs_group')#, promotes = ['*'])
  
        self.add(CoreInputsGroup(
            num_evaluations=num_evaluations,
            num_radial=num_radial,
            num_tangential=num_tangential,
            r_name=r_name,
            tag=tag,
        ), name = 'core_inputs_group')#, promotes=['*'])

        self.add(PreprocessGroup(
            rotor = rotor,
            shape = shape,
            tag = tag,
        ), name = 'preprocess_group')#, promotes = ['*'])

        self.add(AtmosphereGroup(
            rotor = rotor,
            shape = shape,
            mode  = mode,
            tag = tag,
        ), name = 'atmosphere_group')#, promotes = ['*']

        self.add(PhiBracketedSearchGroup(
            rotor = rotor,
            shape = shape,
            mode = mode,
            tag = tag,
        ), name = 'phi_bracketed_search_group')#, promotes = ['*'])

        self.add(InducedVelocityGroup(
            rotor = rotor,
            mode  = mode,
            shape = shape,
            tag = tag,
        ), name = 'induced_velocity_group')#, promotes = ['*'])
            

            
            
