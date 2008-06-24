
import GaudiPython
import unittest

class ConsistencyAlg(GaudiPython.PyAlgorithm):
    
    def init(self, conf ):
        self.conf = conf
        self.esv = g.evtsvc()
        self.esv.dump()
        self.items = {}
        return self
    
    def execute(self):
        """ add a comparison to a persisted repr here maybe in kls __cmp__ ??  """
        loc = self.conf.location()
        print "<%s %s>" % (self.__class__.__name__ , loc )
    
        kco = self.esv[loc]
        assert kco.__class__.__name__ == self.conf.classname()
        if hasattr(kco,"__props__"):
            self.items[len(self)]=kco.__props__()
        print kco
        return True
            
    def __repr__(self):
        return "<%s>" % self.__class__.__name__
    def __getitem__(self,key):
        return key in self.items and self.items[key] or None
    def __len__(self):
        return len(self.items)
        



def _configure():
    import gtconfig
    conf = gtconfig.GenToolsTestConfig( volume="/dd/Geometry/Pool/lvFarPoolIWS" )
    global g
    g = conf.g
    alg = ConsistencyAlg().init(conf)
    g.addAlgorithm(alg)
    return conf



class ConsistencyTestCase(unittest.TestCase):
    """ 
          incompleteness of cleanup means setup/teardown can not be used much                            
    """
    
    def setUp(self):
        self.conf = _configure()

    def tearDown(self):
        pass
        
    def testConsistencyOne(self):
        """ assertions in the execute method of the alg will be triggered if constraints are violated """
        
        print "run test "
        assert self.conf.location() == '/Event/Gen/HepMCEvents'
        assert self.conf.nevents() == 10
        g.run(self.conf.nevents())


    def testConsistencyTwo(self):
        """ assertions in the execute method of the alg will be triggered if constraints are violated """
        
        assert self.conf.location() == '/Event/Gen/HepMCEvents'
        assert self.conf.nevents() == 10
        g.run(self.conf.nevents())




suite = unittest.makeSuite(ConsistencyTestCase,'test')


if __name__ == '__main__':
    #unittest.TextTestRunner(verbosity=2).run(suite)

    conf =  _configure()
    g.run(conf.nevents())
    g.exit()



