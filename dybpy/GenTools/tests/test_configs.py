"""
    StartTime may need to be coordinated for separate runs to be comparable
        
       [dayabaysoft@grid1 tests]$ l */conf.py
-rw-r--r--    1 dayabaysoft dayabay      2958 Jun 26 21:09 829087e9703a05e86287ff82dac0de76/conf.py
-rw-r--r--    1 dayabaysoft dayabay      2958 Jun 26 21:09 a82eaf8bd03e4fefef8006a7e03058c2/conf.py
[dayabaysoft@grid1 tests]$ diff */conf.py
43c43
<                              'StartTime': 3461733009L,
---
>                              'StartTime': 7450269933L, 
         
            HUH VERY DIFFERENT ??
            
            
 nosetests test_configs.py  -s --with-insulate
 nosetests test_configs.py -vvv  -s --with-insulate --with-xml-output --xml-outfile out.xml
  nosetests test_configs.py   -s --with-insulate  --insulate-show-slave-output

 nosetests test_configs.py   -s --with-insulate  --insulate-not-in-slave="--with-xml-output --xml-outfile out.xml"
nosetests test_configs.py   -s --with-insulate  --insulate-not-in-slave="--with-xml-output" --insulate-not-in-slave="--xml-outfile=out.xml"
 
 
    only the succeeding test was reported in the out.xml  ???
           
           
    when one run is bad (segments or whatever) subsequent runs fail too
    ... insulate is not doing the job 
    
                 
                              
      --with-insulate       Enable plugin Insulate: Master insulation plugin class
                        [NOSE_WITH_INSULATE]
  --insulate-skip-after-crash
  --insulate-not-in-slave=NOT_IN_SLAVE
  --insulate-in-slave=IN_SLAVE
  --insulate-show-slave-output
  --with-insulateslave=INSULATESLAVE
  
  
  PROBLEMS :
       not capturing the logged gaudi output only stdout 
  
                       
"""


## generator of (runner,cid) tuples for nosetests consumption


from nose.tools import with_setup
import genrepr
import pprint

def start_t():
    print "start_t ... "    
    
    import ROOT
    ROOT.gSystem.Load("libMathCore")  
    import GaudiPython as gp 
    import PyCintex as pc
    ## 4 lines needed for streams for work ???
    
    from GaudiPython import AppMgr
    g = AppMgr()
    
    """
        attempt to control gaudi messages  ... crashes hard
           void IMessageSvc::setDefaultStream(basic_ostream<char,char_traits<char> >* stream)
         
         ofs = gp.gbl.ofstream()   provides
            <ROOT.basic_ofstream<char,char_traits<char> > object at 0xa64d318>
    
    """
    ims = g.service("MessageSvc",interface="IMessageSvc")
    import sys
    path = "%s.log" % sys.modules[__name__].__file__
    #path = "/dev/stdout"
    log = gp.gbl.ofstream(path)
    ims.setDefaultStream(log)
    
    #from DybTest.gputil import inhibit_run
    #inhibit_run(1)
    import gentools
    
    gen = g.algorithm("GenAlg")
    gen.__class__.__props__ = _gen__props__
    print "start_t completed" 
    
def finish_t():
    print "finish_t ... "
    from GaudiPython import AppMgr
    g = AppMgr()
    g.exit()
    print "finish_t completed "

def setup():
    print "module setup ..... "
    start_t()

def teardown():
    print "module teardown "
    finish_t()

"""
In [12]: g._isvc.CONFIGURED
Out[12]: 1

In [13]: g._isvc.FINALIZED
Out[13]: 2

In [15]: g._isvc.INITIALIZED
Out[15]: 3

In [16]: g._isvc.OFFLINE
Out[16]: 0

"""



class Run:  
    """ 
       use a callable class rather than a function in order to conveniently provide
       a description attribute for nosetests generative test running  
    """
    def __call__(self, cid):
        self.cid = cid
        self.description = self.cid.name()        
        #self.setup = start_t
        #self.teardown = finish_t
        
        from GaudiPython import AppMgr
        g = AppMgr()
        assert not g == None
        state = g.state()
        assert state == 1 , "nope tiz %d " % state 
        
        #g.initialize()
        print g 
        print repr(cid)
        print pprint.pformat(cid.__props__())
        
        g.reinitialize()
        g.run(1)
        
        state = g.state()
        assert state == 3 , "nope tiz %d " % state 
        
        esv = g.evtsvc()
        assert not esv == None
        loc = self.cid['location']
        print "location %s " % loc
        
        ghr = esv[loc]
        assert not ghr == None
        
        event = ghr.event()
        assert not event == None
        
        print repr(ghr)
        print repr(event)
       

# with_setup not working for generators ... set the property on the Run callable
#@with_setup(setup=start_t,teardown=finish_t)
def test_runs():

    from GaudiPython import AppMgr
    g = AppMgr()
    from DybTest import ConfigIdentity
    gen = g.algorithm("GenAlg")
    assert not gen == None  
    
    atts = { 'gen':gen }
    if hasattr(gen,'Location'):
        atts.update( location=gen.Location )
    else:
        atts.update( location='/Event/Gen/GenHeader' )
    
    #yield Run(), ConfigIdentity( "asis" , **atts )

    vs = ['/dd/Geometry/AD/lvAD','/dd/Structure/steel-2/water-2',
          '/dd/Structure/Pool/la-iws','/dd/Geometry/Pool/lvFarPoolIWS','/dd/Structure/AD/la-gds1']
          
    volume = vs[-1] 
    poser = g.property("ToolSvc.GtPositionerTool")
    poser.Volume = volume
    trans = g.property("ToolSvc.GtTransformTool")
    trans.Volume = volume
    
    #yield Run(), ConfigIdentity( "modified volume" , **atts )

    gen.GenTools = [ "GtGunGenTool", "GtTimeratorTool" ]
    #yield Run(), ConfigIdentity( "remove position dd dependent tools" , **atts )

    gsq = g.algorithm("GenSeq")
    dmp = "GtHepMCDumper/GenDump"
    if dmp in gsq.Members:
        g.removeAlgorithm(dmp)
        
    yield Run(), ConfigIdentity( "remove position dd dependent tools, without dumper" , **atts )
    #pass




def _gen__props__(self):
    from GaudiPython import AppMgr
    g = AppMgr()
    assert self.__class__.__name__ == 'iAlgorithm', "wrong class name %s " % self.__class__.__name__
    assert self.name() == "GenAlg" , "wrong instance name %s " % self.name()
    d = {}
    for p in ["Location","GenName","GenTools"]:
        if hasattr(self,p):
            d[p] = getattr(self, p )
    for t in self.GenTools:
        tool = g.toolsvc().create(t)          #tool = g.property("ToolSvc.%s" % t)
        d[t] = {}
        for k,v in tool.properties().items():
            if k not in ['OutputLevel','StartTime']:
                d[t][k] = v.value()
    return d




if __name__=='__main__':
    setup()
    for r,cid in test_runs():
        r(cid)
    teardown()



        
        


