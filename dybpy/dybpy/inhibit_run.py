"""
    this inhibits a number of GaudiPython.AppMgr.run calls (usually one)  
    before replacing this member function  
    
    this allows to import a module that does g.run(n) without invoking the 
    run allowing some further condifuration before really starting the run

"""

from GaudiPython import AppMgr ; g = AppMgr()
_run_inhibit  = 1
_run_original = AppMgr.run

def _control_run(self,nevt):
    """ this prevents g.run(n) from doing so """ 
    
    assert self.__class__.__name__ == 'AppMgr'
    global _run_inhibit
    global _run_original
    
    assert not self.__class__.run == _run_original , "this should never be invoked after the inhibit is lifted "
    
    if _run_inhibit > 0: 
        print "_control_run inhibiting run %s nevt %s _run_inhibit %s " % ( self, nevt, _run_inhibit )
        _run_inhibit -= 1
    else:
        self.__class__.run = _run_original
        print "_control_run replace original run %s nevt %s _run_inhibit %s  " % ( self, nevt, _run_inhibit )
        self.run(nevt)
            
g.__class__.run = _control_run


