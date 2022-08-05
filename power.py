import numpy as np 
import csdl
import csdl_om

# create power vectors with respect to time

class Power(csdl.Model):
    def initialize(self):
        self.parameters.declare('num')
    def define(self):
        num = self.parameters['num']
        
        e_lift = self.declare_variable('e_lift', shape=(num+1,))
        e_cruise = self.declare_variable('e_cruise', shape=(num+1,))
        
        lift_power = self.create_output('lift_power', shape=(num,))
        cruise_power = self.create_output('cruise_power', shape=(num,))
        
        for i in range(0,num):
            lift_power[i,] = (e_lift[i+1] - e_lift[i])*1000000
            cruise_power[i,] = (e_cruise[i+1] - e_cruise[i])*1000000