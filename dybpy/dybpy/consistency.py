
import GaudiPython
import pyutil
g = GaudiPython.AppMgr()


class ConsistencyAlg(GaudiPython.PyAlgorithm, pyutil.PrintLogger):
    
    def init(self, conf ):
        self.conf = conf
        self.esv = g.evtsvc()
        self.esv.dump()
        self.reset()
        return self
    
    def reset(self):
        self.log("reset items ")
        self.items = []
        self.log("after reset " , len=len(self.items) )
    
    def report(self, a , b ):
        import pprint
        return "\n".join( ["",pprint.pformat( a ) ," -------------- compared to ------------------ ", pprint.pformat( b )  ])
    
    def execute(self):
        """
            this is called by the gaudi eventloop machinery
            assertions in here are swallowed by gaudi, and they 
            do not surface as errors/fails but the eventloop gets stopped
        """
        n = len(self.items)
        loc = self.conf.location()
        self.log("execute ", loc=loc , n=n)
        
        kco = self.esv[loc]
        assert kco.__class__.__name__ == self.conf.classname()
        assert hasattr(kco,"__props__")
        #print repr(kco)
        
        self.items.append( kco.__props__() )   
        self.log( "appending item making total " , tot=len(self.items) )                          
        return True
    
             
    def compare(self):
        self.log("comparing ... ", items=len(self.items) )
        for n,itm in enumerate(self.items):
            prior = self.conf.load(n)
            if prior==None:
                self.log("skip comparison %s as no prior" % n)
            else:
                assert prior == itm , " inconsistent event [%s] properties prior/current :  %s " % (n , self.report( prior, itm ))
     
    def save(self):
        self.log( "saving props %d to file " %  len(self.items) )
        for n,itm in enumerate(self.items):
            self.log("save item %s " % n) 
            self.conf.save(n,itm)
                                                                                                                                            
    def __repr__(self):
        return self.hdr()




