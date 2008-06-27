"""
    importing this module interposes customized methods for
        __repr__
        __str__
    into the below customized classes
    
    The below functions are interposed as __props__ methods into the 
    correspondingly named classes, that supply  a dict based structure 
    of the properties of the object 
    used by the __repr__

"""


import DybTest.gputil as gputil


import ROOT
import PyCintex as pc

#pc.loadDictionary("libBaseEventDict")
pc.loadDictionary("libGenEventDict")
#pc.loadDictionary("libSimEventDict")
pc.loadDictionary("libHepMCRflx")
pc.loadDictionary("libCLHEPRflx")

def reload_():
    import sys
    reload(sys.modules[__name__])

def _hdr(self):
    return { '_class':self.__class__.__name__ }


def _CLHEP__HepLorentzVector(self):
    assert self.__class__.__name__ == 'CLHEP::HepLorentzVector'
    d = _hdr(self)
    d.update( px=self.px() , py=self.py() , pz=self.pz() , e=self.e() )
    return d


def _HepMC__GenVertex(self):
    assert self.__class__.__name__ == 'HepMC::GenVertex'
    d = _hdr(self) 
    d.update( position=_CLHEP__HepLorentzVector(self.position()) )
    return d


def _HepMC__GenParticle(self):
    assert self.__class__.__name__ == 'HepMC::GenParticle'
    d = _hdr(self)
    d.update( 
              pdg_id=self.pdg_id() ,
            momentum=_CLHEP__HepLorentzVector(self.momentum()) ,
   production_vertex=_HepMC__GenVertex(self.production_vertex()) 
           )
    return d


def _HepMC__GenEvent(self):
    assert self.__class__.__name__ == 'HepMC::GenEvent'
    d = _hdr(self)
    d.update( event_number=self.event_number() )
    
    particles = []
    for prt in gputil.irange(self.particles_begin(),self.particles_end()):
        particles.append( _HepMC__GenParticle(prt) )
    d.update( particles=particles )
    
    vertices = []
    for vtx in gputil.irange(self.vertices_begin(),self.vertices_end()):
        vertices.append( _HepMC__GenVertex(vtx) )
    d.update( vertices=vertices )
    
    return d


def _DayaBay__HepMCEvent(self):

    assert self.__class__.__name__ == 'DayaBay::HepMCEvent'
    d = _hdr(self)
    d.update( 
        generatorName=self.generatorName() , 
        event=_HepMC__GenEvent(self.event()) 
        )
    return d

    
        
def _DayaBay_GenHeader(self):
    """
        introspective method calling can be dangerous !!
        ... hitting the "release" method decrements the ref count causing the 
        count down to segmentation problem #49
              
              dybgaudi/InstallArea/include/Event/HepMCEvent.h
              gaudi/GaudiKernel/GaudiKernel/KeyedObject.h
    """
    assert self.__class__.__name__ == 'DayaBay::GenHeader'
    d = _hdr(self)
    
    skips = { 
                'serialize':"too complex",
               'fillStream':"handeled in str ",
             'inputHeaders':"too complex",
                  'linkMgr':"too complex" , 
                  'release':"causes decrement of ref count ... countdown to segmentation error" ,
                 'earliest':"prevents consistency", 
                   'latest':"prevents consistency" ,
            }
                
    times = [ 'earliest','latest','timeStamp' ]
    
    meths = [x for x in dir(self) if callable(getattr(self,x))]
    for meth in meths:
        if meth[0:3] not in ['add','set'] and meth not in skips and not meth[0].isupper() and not meth[0] == "_" :
            if meth == "event":
                d[meth] = _HepMC__GenEvent( self.event() )
            elif meth in times:
                its = ROOT.TimeStamp()
                its = getattr(self,meth)()
                d[meth] = its.AsString()
                del its
            else:
                r = getattr(self , meth )()
                d[meth]=repr(r) 
    return d
          
 
def _TimeStamp(self):
    assert self.__class__.__name__ == 'TimeStamp'
    d = _hdr(self)
    return d
                                    
        
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




gputil.dress_classes( 
   {
     'DayaBay::HepMCEvent':_DayaBay__HepMCEvent,
         'HepMC::GenEvent':_HepMC__GenEvent,
      'HepMC::GenParticle':_HepMC__GenParticle,
        'HepMC::GenVertex':_HepMC__GenVertex,
 'CLHEP::HepLorentzVector':_CLHEP__HepLorentzVector,
     'KeyedContainer<DayaBay::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >':
               _KeyedContainer_DayaBay__HepMCEvent,
        'DayaBay::GenHeader':_DayaBay_GenHeader ,
   }
)





