# calculate air density for a given altitude
# valid through 11km altitude
import csdl
import csdl_om


class Density(csdl.Model):
    def initialize(self):
        self.parameters.declare('out_name', types=str)
        self.parameters.declare('alt_name', types=str)
        
    def define(self):
        out_name = self.parameters['out_name']
        alt_name = self.parameters['alt_name']
        
        a1 = -6.5E-3 # K/m
        h1 = 0 # sea level
        h = self.declare_variable(alt_name)
        T_1 = 288.16 # degrees K
        rho_1 = 1.225 # kg/m^3
        g0 = 9.8 # m/s^2
        R = 287
        # compute temperature
        T = T_1 + a1*(h - h1)
        # compute density
        rho = rho_1*(T/T_1)**(-((g0/(a1*R))+1))
        # register output
        self.register_output(out_name, rho)
        
        
        
        
"""
# for testing
class Run(csdl.Model):
    def define(self):
        self.create_input('z', val=5000)
        self.add(Density(out_name='rho', alt_name='z'))


# for testing
sim = csdl_om.Simulator(Run())
sim.run()
print(sim['rho'])
"""







