
import GaudiPython
import unittest
import pyutil
import consistency 

import gtconfig
g = GaudiPython.AppMgr()
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
        gttc = gtconfig.GenToolsTestConfig(alg,volume="/dd/Geometry/Pool/lvFarPoolIWS")
        return gttc


class ConsistencyTestCase(unittest.TestCase, pyutil.PrintLogger):
    """ 
          incompleteness of cleanup means setup/teardown can not be used much                            
    """
    
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
        
    def tearDown(self):
        """
            rewinding is smth to try when operating with event streams from a file
                g.evtsel().rewind()
            but not going to work with generators     
              
        """
        self.log("tearDown %s " % g )
        self.log("alg reset %s " % self.alg )
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



