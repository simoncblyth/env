
"""
   Usage examples :

         import AbtViz.geotree

 


  recursive find and match for TGeoNode and TEveElement

In [1]: tegs.rmatch("logic.*")
Out[1]: 
[<ROOT.TEveGeoShape object ("logicMovalbeTF_0") at 0xbf851a0>,
 <ROOT.TEveGeoShape object ("logicMovalbeTF_0") at 0xbf85340>]

In [2]: tegs.rmatch("Door.*")
Out[2]: [<ROOT.TEveGeoShape object ("Door_0") at 0xbf6ed20>]

In [3]: tegs.rfind("Door_0")
Out[3]: <ROOT.TEveGeoShape object ("Door_0") at 0xbf6ed20>



In [5]: tegs.rselect( lambda self:self.GetRnrSelf() )
Out[5]: 
[<ROOT.TEveGeoShape object ("World_1") at 0xc532230>,
 <ROOT.TEveGeoShape object ("Worldmuon_log_0") at 0xc54a3a8>]


In [1]: tegs.rtagged( avm , 'SKIP' )
Out[1]: 
[<ROOT.TEveGeoShape object ("Door_0") at 0xb569648>,
 <ROOT.TEveGeoShape object ("inner_log_0") at 0xb56a1a0>,
 <ROOT.TEveGeoShape object ("logicMovalbeTF_0") at 0xb57fad0>,
 <ROOT.TEveGeoShape object ("logicMovalbeTF_0") at 0xb57fc70>,
 <ROOT.TEveGeoShape object ("LeadPlateTop_0") at 0xb57fe28>,
 <ROOT.TEveGeoShape object ("LeadPlateBottom_0") at 0xb57ffe0>,
 <ROOT.TEveGeoShape object ("Worldmuon_log_0") at 0xb580198>]




In [7]: gEve.GetEventScene().rfind("EllipticConeSet")
Out[7]: <ROOT.TEveBoxSet object ("EllipticConeSet") at 0xa51be00>


In [8]: gEve.GetEventScene().rmatch(".*")
Out[8]: 
[<ROOT.TEveScene object ("Event scene") at 0xa49d380>,
 <ROOT.TEveEventManager object ("Event") at 0xa641d98>,
 <ROOT.TEveBoxSet object ("EllipticConeSet") at 0xa51be00>,
 <ROOT.TEveStraightLineSet object ("StraightLines") at 0xa4e7e68>]




"""
import re

class irange(object):
    """  cranking the iterator from py side   """
    def __init__(self, begin, end):
        self.begin, self.end = begin, end
    def __iter__(self):
        it = self.begin
        while it != self.end:
            yield it.__deref__()
            it.__postinc__(1)

def eetreelist( ee , path ):
    p = "%s/%s" % ( path , ee.GetName() )
    print "eetreelist %s " % p
    if ee.HasChildren():  
        for child in irange(ee.BeginChildren(),ee.EndChildren()):eetreelist(child, p)



from ROOT import TEveElement
def TEveElement_rfind( self , patn , leaf=True ):
    """
       Many containers have the same names  as their leaves ...
       so introduce the leaf option 

    """
    l = self.rmatch( patn , leaf )
    if len(l)>1:
        print "warning returning first of %d " % len(l)
        for x in l:
            print x
    if len(l)>0:return l[0]
    return None

def TEveElement_rmatch( self , patn , leaf=True ):
    if type(patn) == str:patn = re.compile(patn)
    l = []
    if leaf and self.HasChildren():
        pass 
    else:
        if patn.match(self.GetName()):l.extend([self])  
    if self.HasChildren():
        for child in irange(self.BeginChildren(),self.EndChildren()):l.extend(TEveElement_rmatch(child, patn, leaf ))
    return l

def TEveElement_rselect(self , func ):
    l = []
    if func(self):l.extend([self])  
    if self.HasChildren():
        for child in irange(self.BeginChildren(),self.EndChildren()):l.extend(TEveElement_rselect(child, func))
    return l

def TEveElement_rtagged(self , matcher, tag ):
    l = []
    matcher( self.GetName() )
    if matcher.get( 'pname', None) == tag:l.extend([self])  
    if self.HasChildren():
        for child in irange(self.BeginChildren(),self.EndChildren()):l.extend(TEveElement_rtagged(child, matcher, tag))
    return l

TEveElement.rfind  = TEveElement_rfind
TEveElement.rmatch = TEveElement_rmatch
TEveElement.rselect = TEveElement_rselect
TEveElement.rtagged = TEveElement_rtagged



def test_r():
    """

    2 "logicMovalbeTF_0" volumes with same name :

warning returning first of 2 
<ROOT.TEveGeoShape object ("logicMovalbeTF_0") at 0xa8f1e58>
<ROOT.TEveGeoShape object ("logicMovalbeTF_0") at 0xa8f1ff8>
warning returning first of 2 
<ROOT.TEveGeoShape object ("logicMovalbeTF_0") at 0xa8f1e58>
<ROOT.TEveGeoShape object ("logicMovalbeTF_0") at 0xa8f1ff8>

    """
    from ROOT import gEve
    gs = gEve.GetGlobalScene()
    for v in gs.rmatch(".*"):
        name = v.GetName()
        vv = gs.rfind("%s$" % name)
        if vv == None:
            print "failed to find %s " % name



