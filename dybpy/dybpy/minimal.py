"""
   test setup notes
       1) keep simple short tests
       2) let the test runner catch exceptions..
       3) keeping differnt flavors of nose tests in separate modules
          makes the order in which they are run more predictable
                functional
                doctest
                classbased    
  
    nosetests minimal.py --with-xml-output --xml-outfile=out.xml

"""

# workaround for GaudiPython issue   "class _global_cpp has no attribute 'stringstream'"
import ROOT
ROOT.gSystem.Load("libMathCore")


g = None
gen = None
evt = None


def test_entry():
    from GaudiPython import AppMgr
    global g
    assert g == None
    g = AppMgr()

def test_configure():
    global g
    import inhibit_run
    import gentools    
    global gen
    gen = g.algorithm("GenAlg")
    gen.GenTools = [ "GtGunGenTool", "GtTimeratorTool" ]

def test_run():
    global g
    g.run(1)

def test_evtsvc():   
    global g
    esv = g.evtsvc()    
    global gen
    global evt
    evt = esv[gen.Location]
    assert not evt == None
 
def test_repr_customization():
    import genrepr

def test_str():
    global evt
    for i in range(10):
        print i, str(evt)
  
def test_repr():
    global evt
    for i in range(10):
        print i, repr(evt)
  
test_repr.__test__ = False
   
def test_exit():
    global g
    g.exit()
    assert g.state() not in [1,3],  "appmgr state not cleaned up %d    " % g.state() 
    

if __name__=='__main__':
    print " this is desiged to be run by the nosetests test runner "


