import csdl


class Transition_Efficiency(csdl.Model):
    def initialize(self):
        self.parameters.declare('num')
        self.parameters.declare('m')
    def define(self):
        num = self.parameters['num']
        m = self.parameters['m']
        
        u = self.declare_variable('u', shape=(num+1,))
        w = self.declare_variable('w', shape=(num+1,))
        theta = self.declare_variable('theta', shape=(num+1,))
        z = self.declare_variable('z', shape=(num+1,))
        e_lift = self.declare_variable('e_lift', shape=(num+1,))
        e_cruise = self.declare_variable('e_cruise', shape=(num+1,))
        
        
        dz = u*csdl.sin(theta) - w*csdl.cos(theta)
        final_dz = dz[-1]
        
        dx = u*csdl.cos(theta) + w*csdl.sin(theta)
        final_dx = dx[-1]
        
        final_kinetic_energy = (1/2)*m*(final_dz**2 + final_dx**2)
        
        g = 9.81
        final_potential_energy = m*g*(z[-1] - z[0])
        
        total_energy = (e_lift[-1] + e_cruise[-1])*1000000
        
        e = (final_kinetic_energy + final_potential_energy)/total_energy
        
        self.register_output('e', e)