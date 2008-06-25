
import GaudiPython
import unittest
import pyutil
import consistency 

import gtconfig
g = GaudiPython.AppMgr()
gttc = None 

def _configure():
    global gttc
    if gttc:
        print "_configure reusing gttc "
        return gttc
    else:
        print "_configure instantiating gttc "
        alg = consistency.ConsistencyAlg()
        gttc = gtconfig.GenToolsTestConfig(alg,volume="/dd/Geometry/Pool/lvFarPoolIWS")
        return gttc


class ConsistencyTestCase(unittest.TestCase, pyutil.PrintLogger):
    """ 
          incompleteness of cleanup means setup/teardown can not be used much                            
    """
    
    def setUp(self):
        self.log("setUp")  
        self.conf = _configure()
        self.alg = self.conf.algs['ConsistencyAlg']
        assert issubclass(self.alg.__class__, GaudiPython.PyAlgorithm ) 
        
    def tearDown(self):
        self.log("tearDown")
        self.alg.reset()
        
    def testConsistencyOne(self):
        """ 
             testConsistencyOne
        """
        
        self.log("testConsistencyOne")
        assert self.conf.location() == '/Event/Gen/HepMCEvents'
        assert self.conf.nevents() == 10
        global g
        g.run(self.conf.nevents())

        self.alg.compare()
        self.alg.save()


    def testConsistencyTwo(self):
        """ 
             testConsistencyTwo 
        """
        self.log("testConsistencyTwo> " )
        assert self.conf.location() == '/Event/Gen/HepMCEvents'
        assert self.conf.nevents() == 10
        global g
        g.run(self.conf.nevents())

        self.alg.compare()
        self.alg.save()

suite = unittest.makeSuite(ConsistencyTestCase,'test')

def simple():
    g.run(gttc.nevents())
    g.exit()

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite)



