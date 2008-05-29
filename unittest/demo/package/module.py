"""
module docstring 
"""

from context import whereami as whereami
whereami(globals())


def fail_func_test():
    assert False
    
def pass_func_test():
    pass
    
class module_class:
    """ module_class docstring """
    def __init__(self):
        whereami(globals())
    def fail_method_test(self):
        assert False
    def pass_method_test(self):
        pass

class module_class_test:
    """ module_class_test docstring """
    def __init__(self):
        whereami(globals())
    def fail_method_test(self):
        assert False
    def pass_method_test(self):
        pass



print "=====> instantiating module_class "
mc = module_class()