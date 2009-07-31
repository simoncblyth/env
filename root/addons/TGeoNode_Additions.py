"""

In [3]: tgn.rselect( lambda self:self.IsVisible() )
Out[3]: 
[<ROOT.TGeoNodeMatrix object ("Door_0") at 0x9bdb1d0>,
 <ROOT.TGeoNodeMatrix object ("Lowermuon_log_0") at 0x9c28008>,
 <ROOT.TGeoNodeMatrix object ("top_film_log_0") at 0x9c28530>,
 <ROOT.TGeoNodeMatrix object ("bot_film_log_0") at 0x9c288e0>,
  ...


"""

import re
from ROOT import TGeoNode
def TGeoNode_rfind( self , name ):
    if self.GetName() == name: return self
    kids = self.GetVolume().GetNodes()
    if kids:
        for child in kids:return TGeoNode_rfind(child, name )
def TGeoNode_rmatch( self , patn ):
    if type(patn) == str:patn = re.compile(patn)
    l = []
    if patn.match(self.GetName()):l.extend([self])  
    kids = self.GetVolume().GetNodes()
    if kids:
        for child in kids:l.extend(TGeoNode_rmatch(child, patn))
    return l
def TGeoNode_rselect( self , func ):
    l = []
    if func(self):l.extend([self])  
    kids = self.GetVolume().GetNodes()
    if kids:
        for child in kids:l.extend(TGeoNode_rselect(child, func))
    return l
def TGeoNode_rtagged(self , matcher, tag ):
    l = []
    matcher( self.GetName() )
    if matcher.get( 'pname', None) == tag:l.extend([self])  
    kids = self.GetVolume().GetNodes()
    if kids:
        for child in kids:l.extend(TGeoNode_rtagged(child, matcher, tag))
    return l

TGeoNode.rfind  = TGeoNode_rfind
TGeoNode.rmatch = TGeoNode_rmatch
TGeoNode.rselect = TGeoNode_rselect
TGeoNode.rtagged = TGeoNode_rtagged


