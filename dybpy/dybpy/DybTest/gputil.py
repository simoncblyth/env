"""
    This module provides the machinery for dressing of classes ... 
    by adding/replacing the __repr__ methods, to get the classes to be 
    more amenable interactively and adding a __props__ method to 
    represent instances as simple dicts 
    
       http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/EventIntrospection
   
    Useful slides from Thomas Ruf :
    
       http://lhcb-reconstruction.web.cern.ch/lhcb-reconstruction/Python/GaudiPython_and_RawData.pdf 
       http://lhcb-reconstruction.web.cern.ch/lhcb-reconstruction/Python/Dst_as_Ntuple_files/frame.htm
      
    libMathCore is loaded as a workaround for
           http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/42

    TODO:
        the default repr provides the address of the object 
        ... this should also be given in the replacement  ???
  
          
     methods to check for useful output
              StreamBuffer& KeyedObject<int>::serialize(StreamBuffer& s)
                          
"""

import ROOT
ROOT.gSystem.Load("libMathCore")  
import GaudiPython as gp 
import PyCintex as pc


def reload_():
    import sys
    reload(sys.modules[__name__])


def docs_(kls):
    docs = [getattr(kls,d).__doc__  for d in dir(kls)]
    return docs


def __repr__(self):
    """ pretty print a dict of the properties of the object 
        if an __props__ method that returns the dict has been provided by 
        or added to the class  
    """
    if hasattr(self,"__props__"):
        return format_(self.__props__())
    else:
        return "<%s customized __repr__ not available due to lack of __props__ method  >" % self.__class__.__name__ 

def __str__(self):
    """ invokes the print or fillstream methods of the object if they have them 
        to see this with newlines honoured use "print obj" rather than "str(obj)"
    """
    s = []
    s.append( "<%s >" % self.__class__.__name__ )
    pr = print_(self)
    if pr:
        s.append( pr )
    fs = fillStream_(self)
    if fs:
        s.append( fs )
    return "\n".join(s)

                                    
def dress_classes( klasses ):
    """ replace the repr str methods of the classes """
    for kln,prp in klasses.items(): 
        kls = pc.makeClass(kln)
        kls.__props__ = prp
        kls.__repr__  = __repr__
        kls.__str__   = __str__


def undress_classes( klasses ):
    """ may be unneeded ...
        call the base repr like so 
           object.__repr__(obj)
    """
    for kln,prp in klasses.items(): 
        kls = pc.makeClass(kln)
        if hasattr(kls,'__props__'):
            del kls.__props__ 
        if hasattr(kls,'__repr__'):
            del kls.__repr__ 
        if hasattr(kls,'__str__'):
            del kls.__str__ 
    
    
    
def hdr_(self):
    """ how to access the address of the object on the C++ side ?? """
    d = {}
    d.update( _class=self.__class__.__name__ )    
    return d



class irange(object):
    """  cranking the iterator from py side   TODO: yield a count in a tuple with the item  """
    def __init__(self, begin, end):
        self.begin, self.end = begin, end
        self.count = 0
    def __iter__(self):
        it = self.begin
        while it != self.end:
            yield it.__deref__()
            it.__postinc__(1)
            self.count += 1

def print_(o):
    """
         special handling to call print methods like :
             void GenParticle::print(basic_ostream<char,char_traits<char> >& ostr = std::cout)
          as print is a reserved word in python and as the output goes to a stream 
    """
    if hasattr(o,"print"):
        ss = gp.gbl.stringstream()
        __print = getattr(o, "print")
        __print(ss)
        return ss.str()
    return None


def fillStream_(o):    
    if hasattr(o,"fillStream"):
        ss = gp.gbl.stringstream()
        __fillStream = getattr(o, "fillStream")
        __fillStream(ss)
        return ss.str()
    return None



"""
    this inhibits a number of GaudiPython.AppMgr.run calls (usually one)  
    before replacing this member function  
    
    this allows to import a module that does g.run(n) without invoking the 
    run allowing some further condifuration before really starting the run

"""

_run_inhibit  = 1
_run_original = None

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

def inhibit_run(n=1):

    from GaudiPython import AppMgr 
    global _run_original
    _run_original = AppMgr.run  
    AppMgr.run = _control_run
        
    global _run_inhibit
    _run_inhibit = n
    

def format_(o):
    import pprint
    return pprint.pformat(o)


class PrintLogger:
    def hdr(self):
        return "<%s [0x%08X] > " % ( self.__class__.__name__ , id(self) )
    def log(self, *args , **kwargs ):
        import pprint
        print "%s %s %s" % ( self.hdr() , " ".join(args),  pprint.pformat(kwargs) )







