"""
  dress the appmgr itself

   want to find where the default repr is coming from and put that in the str
   to have access to the address ... but have yet to find where thats done, possibly
   is python builtin 

"""



import GaudiPython 

def __repr__(self):
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


#def __str__



GaudiPython.Bindings.AppMgr.__repr__=__repr__
