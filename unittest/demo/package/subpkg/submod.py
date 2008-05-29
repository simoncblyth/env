"""
  subpkg/submod.py docstring
"""

from context import whereami as whereami
whereami(globals())


def fail_func_test():
    assert False
    
def pass_func_test():
    pass
    
class submod_class:
    """ submod_class docstring """
    def __init__(self):
        whereami(globals())
    def fail_method_test(self):
        assert False
    def pass_method_test(self):
        pass


submod_class_test=submod_class

#class submod_class_test:
#    """ moda_class_test docstring """
#    def __init__(self):
#        whereami(globals())
#    def fail_method_test(self):
#        assert False
#    def pass_method_test(self):
#        pass



print "=====> instantiating submod_class "
smc = submod_class()

print "=====> instantiating submod_class_test "
smct = submod_class_test()



