"""
    This module provides some dressing of classes ... 
    by adding/replacing the __repr__ methods, to get the classes to be 
    more amenable interactively and adding a __props__ method to 
    represent instances as simple dicts 
    
    next steps
        ... support for picking dicts, marshalling the pickles 
           http://www.ibm.com/developerworks/library/l-pypers.html
        ... cmp operators  
    
    
       http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/EventIntrospection
   
    Useful slides from Thomas Ruf :
    
       http://lhcb-reconstruction.web.cern.ch/lhcb-reconstruction/Python/GaudiPython_and_RawData.pdf 
       http://lhcb-reconstruction.web.cern.ch/lhcb-reconstruction/Python/Dst_as_Ntuple_files/frame.htm
      
   CAUTION : 
      many of the below are repr methods to be interposed into the classes of interest
      ... so every self is a different self

    libMathCore is loaded as a workaround for
           http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/42



    TODO:
        the default repr provides the address of the object 
        ... this should also be given in the replacement 


"""

import ROOT
ROOT.gSystem.Load("libMathCore")  
#import GaudiPython as gp 
import gputil

from PyCintex import *
#loadDictionary("libBaseEventDict")
loadDictionary("libGenEventDict")
#loadDictionary("libSimEventDict")
loadDictionary("libHepMCRflx")
loadDictionary("libCLHEPRflx")


def __repr__(self):
    return gputil.format_(self.props)

def _hdr(self):
    """ how to access the address of the object on the C++ side ?? """
    self.props = d = {}
    d.update( _class=self.__class__.__name__ )
    pr = gputil.print_(self)
    if pr:
        d.update( _print=pr )
    fs = gputil.fillStream_(self)
    if fs:
        d.update( _fillStream=fs )
    return d


def _CLHEP__HepLorentzVector(self):
    assert self.__class__.__name__ == 'CLHEP::HepLorentzVector'
    d = _hdr(self)
    d.update( px=self.px() , py=self.py() , pz=self.pz() , e=self.e() )
    return d
#def _repr_CLHEP__HepLorentzVector(self):
#    return gputil.format_(_CLHEP__HepLorentzVector(self))


def _HepMC__GenVertex(self):
    assert self.__class__.__name__ == 'HepMC::GenVertex'
    d = _hdr(self) 
    d.update( position=_CLHEP__HepLorentzVector(self.position()) )
    return d
#def _repr_HepMC__GenVertex(self):
#    return _format(_HepMC__GenVertex(self))




def _HepMC__GenParticle(self):
    assert self.__class__.__name__ == 'HepMC::GenParticle'
    d = _hdr(self)
    d.update( 
              pdg_id=self.pdg_id() ,
            momentum=_CLHEP__HepLorentzVector(self.momentum()) ,
   production_vertex=_HepMC__GenVertex(self.production_vertex()) 
           )
    return d
#def _repr_HepMC__GenParticle(self):
#    return _format(_HepMC__GenParticle(self))





def _HepMC__GenEvent(self):
    assert self.__class__.__name__ == 'HepMC::GenEvent'
    d = _hdr(self)
    d.update( event_number=self.event_number() )
    
    particles = []
    for prt in irange(self.particles_begin(),self.particles_end()):
        particles.append( _HepMC__GenParticle(prt) )
    d.update( particles=particles )
    
    vertices = []
    for vtx in irange(self.vertices_begin(),self.vertices_end()):
        vertices.append( _HepMC__GenVertex(vtx) )
    d.update( vertices=vertices )
    
    return d
#def _repr_HepMC__GenEvent(self):
#    return _format(_HepMC__GenEvent(self))



def _DayaBay__HepMCEvent(self):
    """
        methods to check for useful output
              StreamBuffer& KeyedObject<int>::serialize(StreamBuffer& s)
    
    """
    assert self.__class__.__name__ == 'DayaBay::HepMCEvent'
    d = _hdr(self)
    d.update( 
        generatorName=self.generatorName() , 
        event=_HepMC__GenEvent(self.event()) 
        )
    return d
#def _repr_DayaBay__HepMCEvent(self):
#    return _format(_DayaBay_HepMCEvent(self))
    
    
        
def _KeyedContainer_DayaBay__HepMCEvent(self):
    assert self.__class__.__name__ == 'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >'
    d = _hdr(self)
    d.update( clID=self.clID() , name=self.name() , len=len(self) )
    child = []
    for itm in self:
        assert itm.parent() == self
        child.append( _DayaBay__HepMCEvent(itm) )
    d.update( child=child )
    return d
#def _repr_KeyedContainer_DayaBay__HepMCEvent(self):
#    return _format(_KeyedContainer_DayaBay__HepMCEvent(self))





                
                        
def dress_classes( klasses ):
    """ provide the classes with shiny new repr for interactive ease """
    for kln,prp in klasses.items(): 
        kls = makeClass(kln)
        kls.__props__ = prp
        kls.__repr__  = __repr__


"""    
dress_classes( 
   {
     'DayaBay::HepMCEvent':(_repr_DayaBay__HepMCEvent,_DayaBay__HepMCEvent),
         'HepMC::GenEvent':(_repr_HepMC__GenEvent,_HepMC__GenEvent),
      'HepMC::GenParticle':(_repr_HepMC__GenParticle,_HepMC__GenParticle),
        'HepMC::GenVertex':(_repr_HepMC__GenVertex,_HepMC__GenVertex),
 'CLHEP::HepLorentzVector':(_repr_CLHEP__HepLorentzVector,_CLHEP__HepLorentzVector),
     'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >':
               (_repr_KeyedContainer_DayaBay__HepMCEvent,_KeyedContainer_DayaBay__HepMCEvent),
   }
)
"""

dress_classes( 
   {
     'DayaBay::HepMCEvent':_DayaBay__HepMCEvent,
         'HepMC::GenEvent':_HepMC__GenEvent,
      'HepMC::GenParticle':_HepMC__GenParticle,
        'HepMC::GenVertex':_HepMC__GenVertex,
 'CLHEP::HepLorentzVector':_CLHEP__HepLorentzVector,
     'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >':
               _KeyedContainer_DayaBay__HepMCEvent,
   }
)





