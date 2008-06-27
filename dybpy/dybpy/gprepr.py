"""
  dress the appmgr itself

   want to find where the default repr is coming from and put that in the str
   to have access to the address ... but have yet to find where thats done, possibly
   is method of object base class 

"""



import GaudiPython 
import pprint 

def hdr(self):
    return "<%s> [0x%08X] " % ( self.__class__.__name__ , id(self) )


def AppMgr__repr__(self):
    """
        huh sometimes class name if AppMgr 
    """
    assert self.__class__.__name__ in ['GaudiPython.Bindings.AppMgr','AppMgr'], "invalid classname %s " % repr(self.__class__.__name__)
    
    d = []
    d.append("<%s> [%s] state:%s " % (self.__class__.__name__ , id(self) , self.state() ))
    d.append("TopAlg... [%s] " % len(self.TopAlg) ) 
    for a in self.TopAlg:
        d.append(a)
    return " ".join(d)


def iProperty__repr__(self):
    d = {}
    for k,v in self.properties().items():
        d[k] = v.value()
    return "\n".join([hdr(self),pprint.pformat(d)])






GaudiPython.Bindings.AppMgr.__repr__ = AppMgr__repr__
#GaudiPython.Bindings.iAlgTool.__repr__ = iAlgTool__repr__
GaudiPython.Bindings.iProperty.__repr__ = iProperty__repr__


def reload_():
    import sys
    reload(sys.modules[__name__])