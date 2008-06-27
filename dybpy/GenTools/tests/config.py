"""
    Following the messy configuration ... 
    establish an identity for the configuation, for use as a persistency context

    Do this by interposing __props__ methods on classes that comprise
    the identity of this configuration and instanciate the ConfigIdentity with them

"""


from GaudiPython import AppMgr
g = AppMgr()
        
from DybTest.gputil import inhibit_run
inhibit_run(1)
import gentools
 
gen = g.algorithm("GenAlg")
#gen.OutputLevel = 1
gen.GenTools = [ "GtGunGenTool", "GtTimeratorTool" ]
## get rid of position and tranform tools as they are not finding the detector element

gsq = g.algorithm("GenSeq")        
dmp = "GtHepMCDumper/GenDump"
if dmp in gsq.Members:
    g.removeAlgorithm(dmp)

print "reset the positioner tool volume " 
#volume = '/dd/Geometry/AD/lvAD'
#volume = '/dd/Structure/steel-2/water-2'
#volume = '/dd/Structure/Pool/la-iws'
#volume="/dd/Geometry/Pool/lvFarPoolIWS"
volume = '/dd/Structure/AD/la-gds1'
        
poser = g.property("ToolSvc.GtPositionerTool")
poser.Volume = volume
trans = g.property("ToolSvc.GtTransformTool")
trans.Volume = volume
        


def _gen__props__(self):
    global g
    assert self.__class__.__name__ == 'iAlgorithm', "wrong class name %s " % self.__class__.__name__
    assert self.name() == "GenAlg" , "wrong instance name %s " % self.name()
    d = {}
    for p in ["Location","GenName","GenTools"]:
        d[p] = getattr(self, p )

    for t in self.GenTools:
        #tool = g.property("ToolSvc.%s" % t)
        tool = g.toolsvc().create(t)
        d[t] = {}
        for k,v in tool.properties().items():
            if k not in ['OutputLevel']:
                d[t][k] = v.value()

    return d

gen.__class__.__props__ = _gen__props__


from DybTest import ConfigIdentity
cid = ConfigIdentity( gen=gen )
    
        
        



"""
#pot = g.toolsvc().create("GtPositionerTool")
#trt = g.toolsvc().create("GtTransformTool")
#trt.Volume = pot.Volume = None
#msv = g.service("MessageSvc")
#msv.OutputLevel = ol
#gun = g.property("ToolSvc.GtGunGenTool")
#gun.OutputLevel = ol

#def cleanup():
#    for alg in g.TopAlg:
#        g.removeAlgorithm(alg)
    
"""



