import csdl


class timestep(csdl.Model):
    def initialize(self):
        self.parameters.declare('num')
    def define(self):
        num = self.parameters['num']
        dt = self.declare_variable('dt')
        h_vec = csdl.expand(dt, num)
        self.register_output('h', h_vec)