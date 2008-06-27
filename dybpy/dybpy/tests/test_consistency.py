import GaudiPython
from GaudiPython import AppMgr 
g = AppMgr()

import unittest
import consistency 
import gtconfig

gttc = None 

def reload_():
    import sys
    reload(sys.modules[__name__])
    
def _configure():
    global gttc
    if gttc:
        print "_configure reusing gttc"
        return gttc
    else:
        print "_configure instantiating gttc "
        alg = consistency.ConsistencyAlg()
        g.EvtMax = 5
        gttc = gtconfig.GenToolsTestConfig(alg)
        return gttc
        
class ConsistencyTestCase(unittest.TestCase, pyutil.PrintLogger):
    def setUp(self):
        """
            previously when were doing double initialize the py algorithm instance  
            after the 2nd initialize was out of step , gaudi clinging on to the initial instance
            note that the PyAlgorithm is not the same as the gaudi side algorithm     
        """
        self.log("setUp %s" % g )
        self.conf = _configure()
        name = "ConsistencyAlg"
        self.alg = self.conf.algs[name]
        assert issubclass( self.alg.__class__, GaudiPython.PyAlgorithm )
        assert issubclass( g.algorithm(name).__class__ , GaudiPython.Bindings.iAlgorithm)
        assert not(self.alg is g.algorithm(name)), " alg consistent %s cf %s " % ( self.alg , g.algorithm(name) )
    
    def testConsistencyOne(self):
        self.log("testConsistencyOne")
        global g
        g.run(g.EvtMax)
        self.alg.compare()
        self.alg.save()
    
    def testConsistencyTwo(self):
        self.log("testConsistencyTwo> " )
        global g
        g.run(g.EvtMax)
        self.alg.compare()
        self.alg.save()
        pass
    testConsistencyTwo.__test__ = False
         
    def tearDown(self):
        self.log("tearDown %s " % g )
        self.log("alg reset %s " % self.alg )
        self.alg.reset()
        


suite = unittest.makeSuite(ConsistencyTestCase,'test')
def simple():
    g.run(gttc.nevents())
    g.exit()
if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite)



