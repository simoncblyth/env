
import GaudiPython
from GaudiPython import AppMgr, PyAlgorithm 
g = AppMgr()

import config
from config import cid as cid


class ConsistencyAlg(PyAlgorithm):
    def log(self, msg ):
        print "%s %s " % ( self.__class__.__name__ , msg )
    
    def init(self, cid):
        self.cid = cid
        self.esv = g.evtsvc()
        self.esv.dump()
        self.reset()
        return self
    
    def reset(self):
        self.log("reset items ")
        self.items = []
    
    def report(self, a , b ):
        import pprint
        return "\n".join( ["",pprint.pformat( a ) ," -------------- compared to ------------------ ", pprint.pformat( b )  ])
    
    def initialize(self):
        self.log("initialize ")
        
    def finalize(self):
        self.log("finalize")
    
    def execute(self):
        """
            this is called by the gaudi eventloop machinery
            assertions in here are swallowed by gaudi, and they 
            do not surface as errors/fails but the eventloop gets stopped
        """
        n = len(self.items)
        loc = self.cid['gen'].Location
        self.log("execute ", loc=loc , n=n)
        kco = self.esv[loc]
        assert hasattr(kco,"__props__")
        self.items.append( kco.__props__() )                           
        return True
    
             
    def compare(self):
        self.log("comparing ... ", items=len(self.items) )
        for n,itm in enumerate(self.items):
            prior = self.cid.load(n)
            if prior==None:
                self.log("skip comparison %s as no prior" % n)
            else:
                assert prior == itm , " inconsistent event [%s] properties prior/current :  %s " % (n , self.report( prior, itm ))
     
    def save(self):
        self.log( "saving props %d to file " %  len(self.items) )
        for n,itm in enumerate(self.items):
            self.log("save item %s " % n) 
            self.cid.save(n,itm)
                                                                                                                                            
    def __repr__(self):
        return self.hdr()



alg = ConsistencyAlg().init(cid)
g.EvtMax = 5
g.addAlgorithm(alg) 
    

import unittest
                                                    
class ConsistencyTestCase(unittest.TestCase):
    def setUp(self):
        global alg
        self.alg = alg
        assert issubclass( self.alg.__class__, GaudiPython.PyAlgorithm )
        assert issubclass( g.algorithm('ConsistencyAlg').__class__ , GaudiPython.Bindings.iAlgorithm)
    
    def testConsistencyOne(self):
        global g
        g.run(g.EvtMax)
        self.alg.compare()
        self.alg.save()
    
    def testConsistencyTwo(self):
        global g
        g.run(g.EvtMax)
        self.alg.compare()
        self.alg.save()
        pass
    testConsistencyTwo.__test__ = False
         
    def tearDown(self):
        self.alg.reset()
        

suite = unittest.makeSuite(ConsistencyTestCase,'test')

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite)



