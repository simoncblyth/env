
import GaudiPython
import unittest
import pyutil

from gtconfig import gttc
g = GaudiPython.AppMgr()


class ConsistencyTestCase(unittest.TestCase, pyutil.PrintLogger):
    """ 
          incompleteness of cleanup means setup/teardown can not be used much                            
    """
    
    def setUp(self):
        self.log("setUp")  
        global gttc
        self.conf = gttc
        
    def tearDown(self):
        self.log("tearDown ... skip cleanup")
        #self.conf.cleanup()
        
    def testConsistencyOne(self):
        """ 
             testConsistencyOne
        """
        
        
        self.log("testConsistencyOne")
        assert self.conf.location() == '/Event/Gen/HepMCEvents'
        assert self.conf.nevents() == 10
        global g
        g.run(self.conf.nevents())

        alg = self.conf.alg
        alg.compare()
        alg.save()


    def testConsistencyTwo(self):
        """ 
             testConsistencyTwo 
        """
        self.log("testConsistencyTwo> " )
        assert self.conf.location() == '/Event/Gen/HepMCEvents'
        assert self.conf.nevents() == 10
        global g
        g.run(self.conf.nevents())

        alg = self.conf.alg
        alg.compare()
        alg.save()


suite = unittest.makeSuite(ConsistencyTestCase,'test')


def simple():
    g.run(gttc.nevents())
    g.exit()

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite)



