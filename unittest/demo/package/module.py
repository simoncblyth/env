"""
module docstring 
"""

import unittest


# these 3 are needed for main running not for use a part of package as this is done in the __init__.py 
import os
import sys
sys.path.append(os.path.join(os.environ['ENV_HOME'], "unittest/context" ) )


from context import ctx as ctx
from context import present as present 

present(ctx(globals()))


fails = False

def setup():
    present(ctx(globals()))
    
def teardown():
    present(ctx(globals()))

def fail_func_test():
    present(ctx(globals()))
    assert False
fail_func_test.__test__ = fails
    
def pass_func_test():
    present(ctx(globals()))
    pass
 
    
                        
class module_class:
    """ module_class docstring """
    def __init__(self):
        present(ctx(self))
    def fail_method_test(self):
        present(ctx(self))
        assert False
    fail_method_test.__test__ = fails
    def pass_method_test(self):
        present(ctx(self))
        pass

class module_class_test:
    """ module_class_test docstring """
    
    def setup(self):
        present(ctx(self))
    def teardown(self):
        present(ctx(self))
    def __init__(self):
        present(ctx(self))
    def fail_method_test(self):
        present(ctx(self))
        assert False
    fail_method_test.__test__ = fails
        
    def pass_method_test(self):
        present(ctx(self))
        pass


class module_class_unit(unittest.TestCase):
    
    #def __init__(self):
    #    """ this fails with ...  
    #         TypeError: __init__() takes exactly 1 argument (2 given)
    #    """
    #    unittest.TestCase.__init__(self)
    
    def __init__(self, *args):
        """ add some boilerplate to pass args """
        unittest.TestCase.__init__(self, *args)
        present(ctx(self))
        
        
    def setup(self):
        """ this does not run the name must be setUp """
        present(ctx(self))
    def teardown(self):
        """ this does not run the name must be tearDown """
        present(ctx(self))
    def setUp(self):
        present(ctx(self))
    def tearDown(self):
        present(ctx(self))
 
    
    def test_fail_method(self):
        present(ctx(self))
        assert False
        
    test_fail_method.__test__ = fails
        
    def test_pass_method(self):
        present(ctx(self))
        pass

print "=====> instantiating module_class "
mc = module_class()


if __name__=='__main__':
    unittest.main()
