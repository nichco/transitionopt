import numpy as np
from csdl import Model
import csdl
from rotor_parameters import RotorParameters
import openmdao.api as om



class AirfoilSurrogateModelGroup(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('shape')#, types = tuple)
        self.parameters.declare('rotor')#, types = RotorParameters)
        self.parameters.declare('tag', types=str)
    
    def define(self):
        shape = self.parameters['shape']
        rotor = self.parameters['rotor']
        tag = self.parameters['tag']
        
        self.add_input('Re'+tag, shape = shape)
        self.add_input('alpha_distribution'+tag, shape = shape)
        # self.add_input('AoA', shape = shape)
        self.add_input('chord_distribution'+tag, shape = shape)
        

        self.add_output('Cl'+tag, shape = shape)
        # self.add_output('Cl_2', shape = shape)
        
        self.add_output('Cd'+tag, shape = shape)
        # self.add_output('Cd_2', shape = shape)

        indices = np.arange(shape[0] * shape[1] * shape[2])
        self.declare_derivatives ('Cl'+tag, 'Re'+tag, rows = indices, cols = indices)
        self.declare_derivatives ('Cl'+tag, 'alpha_distribution'+tag, rows = indices, cols = indices)
        # self.declare_derivatives ('Cl_2', 'Re', rows = indices, cols = indices)
        # self.declare_derivatives ('Cl_2', 'AoA', rows = indices, cols = indices)
        
        self.declare_derivatives ('Cd'+tag, 'Re'+tag, rows = indices, cols = indices)
        self.declare_derivatives ('Cd'+tag, 'alpha_distribution'+tag, rows = indices, cols = indices)
        # self.declare_derivatives ('Cd_2', 'Re', rows = indices, cols = indices)
        # self.declare_derivatives ('Cd_2', 'AoA', rows = indices, cols = indices)

        self.x_1 = np.zeros((shape[0] * shape[1] * shape[2], 2))
        # self.x_2 = np.zeros((shape[0] * shape[1] * shape[2], 2))

        

    def compute(self, inputs, outputs):
        shape       = self.parameters['shape']
        rotor       = self.parameters['rotor']
        tag = self.parameters['tag']
        
        interp      = rotor['interp']

        chord       = inputs['chord_distribution'+tag].flatten()
        alpha       = inputs['alpha_distribution'+tag].flatten()
        Re          = inputs['Re'+tag].flatten()
        # AoA         = inputs['AoA'].flatten()

        self.x_1[:, 0] = alpha
        self.x_1[:, 1] = Re/2e6

        # self.x_2[:, 0] = AoA
        # self.x_2[:, 1] = Re/2e6
 
        y_1 = interp.predict_values(self.x_1).reshape((shape[0] , shape[1] , shape[2], 2))
        # y_2 = interp.predict_values(self.x_2).reshape((shape[0] , shape[1] , shape[2], 2))

        # print( y_1[:,:,:,0])
        outputs['Cl'+tag] = y_1[:,:,:,0]
        outputs['Cd'+tag] = y_1[:,:,:,1]

        # outputs['Cl_2'] = y_2[:,:,:,0]
        # outputs['Cd_2'] = y_2[:,:,:,1]


    def compute_derivatives(self, inputs, derivatives):
        rotor       = self.parameters['rotor']
        tag = self.parameters['tag']
        
        interp      = rotor['interp']
        
        alpha       = inputs['alpha_distribution'+tag].flatten()
        Re          = inputs['Re'+tag].flatten()
        # AoA         = inputs['AoA'].flatten()
       
        self.x_1[:, 0] = alpha
        self.x_1[:, 1] = Re/2e6

        # self.x_2[:, 0] = AoA
        # self.x_2[:, 1] = Re/2e6

        dy_dalpha = interp.predict_derivatives(self.x_1, 0)
        dy_dRe = interp.predict_derivatives(self.x_1, 1)

        derivatives['Cl'+tag, 'alpha_distribution'+tag] = dy_dalpha[:, 0]
        derivatives['Cd'+tag, 'alpha_distribution'+tag] = dy_dalpha[:, 1]

        derivatives['Cl'+tag, 'Re'+tag] = dy_dRe[:, 0] /2e6
        derivatives['Cd'+tag, 'Re'+tag] = dy_dRe[:, 1] /2e6

        # dy_dalpha_2 = interp.predict_derivatives(self.x_2, 0)
        # dy_dRe_2 = interp.predict_derivatives(self.x_2, 1)

        # derivatives['Cl_2', 'AoA'] = dy_dalpha_2[:, 0]
        # derivatives['Cd_2', 'AoA'] = dy_dalpha_2[:, 1]

        # derivatives['Cl_2', 'Re'] = dy_dRe_2[:, 0] /2e6
        # derivatives['Cd_2', 'Re'] = dy_dRe_2[:, 1] /2e6

