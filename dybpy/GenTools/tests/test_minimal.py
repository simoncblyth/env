"""
   This is meant to be run by nosetests with commands such as :
      
         nosetests           
                     sniffs out tests in cwd and beneath
                         
         nosetests minimal.py
                         restrict to a module
       
   useful options :
       -s --no-capture    : allows to see the stdout from tests 
       --help             : long list of options 
       -v / -vv / -vvv    : verbosity control 
       -p                 : list of plugins
         
   
   for automated running from project directory something like 
   the below is used 
       nosetests  --with-xml-output --xml-outfile=out.xml


   test setup notes
       1) keep tests simple and short 
       2) let the test runner catch exceptions..
       3) keeping different flavors of nose tests (functional,doctest,classbased) 
          in separate modules makes the order in which they are run more predictable
  
"""

g = None
gen = None
evt = None


def test_workaround():
    """ workaround for GaudiPython issue   "class _global_cpp has no attribute 'stringstream'" """
    import ROOT
    ROOT.gSystem.Load("libMathCore")

def test_entry():
    from GaudiPython import AppMgr
    global g
    assert g == None
    g = AppMgr()

def test_configure():
    global g
    
    from DybTest.gputil import inhibit_run
    inhibit_run(1)
    
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
 
 
test_repr.__test__ = True    # tests can be switched off
   
def test_exit():
    global g
    g.exit()
    assert g.state() not in [1,3],  "appmgr state not cleaned up %d    " % g.state() 
    

if __name__=='__main__':
    import sys
    print sys.modules[__name__].__doc__


