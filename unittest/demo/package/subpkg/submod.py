"""subpkg/submod.py docstring"""

from context import ctx as ctx
from context import present as present 

from .. import log 

present(ctx(globals()))

import unittest
import sys

fails = False

def setup():
    present(ctx(globals()))
def teardown():
    present(ctx(globals()))
def fail_func_test():
    """ fail_func_test docstring """
    present(ctx(globals()))
    assert False
fail_func_test.__test__=fails
def pass_func_test():
    """ pass_func_test docstring """
    present(ctx(globals()))
    pass
    
class submod_class:
    """ submod_class docstring 
        these tests are not run due to the name of the class not including _test
    """
    def __init__(self):
        present(ctx(self))
    def fail_method_test(self):
        """ fail_method_test docstring """
        assert False
    fail_method_test.__test__=fails
    def pass_method_test(self):
        """ pass_method_test docstring """
        pass

# doing the below does not trick nosetests ... need the full defn 
#submod_class_test=submod_class

class submod_class_test:
    """ submod_class_test docstring """
    def __init__(self):
        present(ctx(self))
    def fail_method_test(self):
        """ fail_method_test docstring"""
        present(ctx(self))
        assert False
    fail_method_test.__test__=fails
    def pass_method_test(self):
        """ pass_method_test docstring """
        present(ctx(self))
        pass


class submod_class_unit(unittest.TestCase):
    """  these tests are run despite the name due to  the TestCase subclassing """
    
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
    def setUp(self):
        present(ctx(self))
    def teardown(self):
        """ this does not run the name must be tearDown """
        present(ctx(self))
    def tearDown(self):
        present(ctx(self))
    def test_fail_method(self): 
        """ test_fail_method docstring """
        present(ctx(self))
        assert False
    test_fail_method.__test__=fails
    def test_pass_method(self): 
        """ test_pass_method docstring """
        present(ctx(self))
        pass



log.debug("=====> instantiating submod_class ")
smc = submod_class()

log.debug("=====> instantiating submod_class_test ")
smct = submod_class_test()



