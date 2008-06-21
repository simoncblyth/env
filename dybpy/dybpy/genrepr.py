"""
    This module provides some dressing of classes ... 
    by adding/replacing the __repr__ methods, to get the classes to be 
    more amenable interactively 
    
       http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/EventIntrospection
   
    Useful slides from Thomas Ruf :
    
       http://lhcb-reconstruction.web.cern.ch/lhcb-reconstruction/Python/GaudiPython_and_RawData.pdf 
       http://lhcb-reconstruction.web.cern.ch/lhcb-reconstruction/Python/Dst_as_Ntuple_files/frame.htm
      
   CAUTION : 
      many of the below are repr methods to be interposed into the classes of interest
      ... so every self is a different self

    libMathCore is loaded as a workaround for
           http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/42

"""

import ROOT
ROOT.gSystem.Load("libMathCore")  

from PyCintex import *
#loadDictionary("libBaseEventDict")
loadDictionary("libGenEventDict")
#loadDictionary("libSimEventDict")
loadDictionary("libHepMCRflx")
loadDictionary("libCLHEPRflx")

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

def _hdr(self):
    return " <%s> " % ( self.__class__.__name__ )

def _KeyedContainer_DayaBay__HepMCEvent(self):
    dmp = []
    dmp.append(_hdr(self) + " clID:%s name:%s len:%s " % (self.clID(), self.name(), len(self) ))
    for itm in self:
        assert itm.__class__.__name__ == 'DayaBay::HepMCEvent' 
        assert itm.parent() == self
        dmp.append( repr(itm) )
    return "\n".join(dmp)

def _DayaBay__HepMCEvent(self):
    dmp = []
    dmp.append(_hdr(self) + " generatorName:%s " % ( self.generatorName() ))
    evt = self.event()
    assert evt.__class__.__name__ == 'HepMC::GenEvent'
    dmp.append( repr(evt) )
    return "\n".join(dmp)
    
def _HepMC__GenEvent(self):
    dmp = []
    dmp.append(_hdr(self) + " event_number:%s" % self.event_number())
    
    for prt in irange(self.particles_begin(),self.particles_end()):
        assert prt.__class__.__name__ == 'HepMC::GenParticle'
        dmp.append( repr(prt) )
    
    for vtx in irange(self.vertices_begin(),self.vertices_end()):
        assert vtx.__class__.__name__ == 'HepMC::GenVertex'
        dmp.append( repr(vtx) )
        
    return "\n".join(dmp)

def _HepMC__GenParticle(self):
    dmp = [] 
    dmp.append(_hdr(self) + " pdg_id:%d " % ( self.pdg_id() ) )
    dmp.append( "         momentum:%s" % repr(self.momentum() ))
    dmp.append(" production_vertex:%s" % repr(self.production_vertex() ))
    return "\n".join(dmp)

def _HepMC__GenVertex(self):
    dmp = []
    dmp.append(_hdr(self))
    dmp.append("position:%s" % repr(self.position() ))
    return "\n".join(dmp)

def _CLHEP__HepLorentzVector(self):
    return _hdr(self) + " (%s,%s,%s;%s) " % (  self.px() , self.py() , self.pz() , self.e() )


def dress_classes( klns ):
    """ provide the classes with shiny new repr for interactive ease """
    for kln in klns:
        name = "_" + kln.replace("::" , "__")
        if name in globals():
            print "dressing kln %s name %s " % ( kln , name ) 
            kls = makeClass(kln)
            kls.__repr__ = globals()[name]
        else:
            print "kln %s name %s not in globals " %  ( kln , name )

def dress_uglies( ugly ):
    for kln,rep in ugly.items(): 
        kls = makeClass(kln)
        kls.__repr__ = rep  


dress_classes( [ 
  'DayaBay::HepMCEvent',
  'HepMC::GenEvent',
  'HepMC::GenParticle' , 
  'HepMC::GenVertex',
  'CLHEP::HepLorentzVector',
   ] )

dress_uglies(
   { 'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >':_KeyedContainer_DayaBay__HepMCEvent }
   )

