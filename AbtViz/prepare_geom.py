import os
import re
import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
from ROOT import kTRUE, kFALSE

# adds recursive tree methods : rfind/rmatch/rselect/rtagged
import env.root.addons.TGeoNode_Additions
import env.root.addons.TEveElement_Additions

def treewrap( egn , path ):
    """  
       Wrap tree of TGeoVolume/TGeoNode  into tree of TEveGeoNode
       NB wrap done in child loop ... so top is not wrapped
    """
    p = "%s/%s" %  ( path , egn.GetName() )
    print "treewrap %s " % p 
    oa = egn.GetNode().GetVolume().GetNodes() 
    if oa:
        for child in oa:
            cegn = TEveGeoNode(child) 
            egn.AddElement( cegn )     # parent hookup 
            treewrap( cegn , p );


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    from ROOT import TGeoManager, gGeoManager
    
    from geoconf import GeoConf, VolMatcher
    TGeoManager.Import(GeoConf.rawpath)
    tgn = gGeoManager.GetTopNode()
    tgv = gGeoManager.GetTopVolume()

    ## coloring and vizibility are set on the TGeoNode/TGeoVolume 

    vmr = VolMatcher()
    from geoedit import VolEditor, TGeoWalk, treelist
    ved = VolEditor()
    tgw = TGeoWalk( vmr , ved )
    tgw( tgn )
    print vmr.report()

    ## traverse the tree dumping the colors and vizibilities
    treelist( tgn , "treelist")

    from ROOT import TEveManager, TEveGeoTopNode, TEveGeoNode, TEveGeoShape, TFile, gEve
    visopt, vislvl, maxvisnds = 1, 9, 10000
    egtn = TEveGeoTopNode( gGeoManager, tgn , visopt , vislvl , maxvisnds )   ## vivlvl is somehow not workin
    treewrap( egtn , "" ) 

    ##
    ##    TEveGeoNode::Save invokes TEveGeoShapeExtract* TEveGeoNode::DumpShapeTree(TEveGeoNode* geon, TEveGeoShapeExtract* parent, Int_t level)
    ##    where the translation from TGeo to TEveGeo is done...
    ##    DumpShapeTree is private ... so have to go via file 
    ## 
    egtn.Save( GeoConf.xpath, GeoConf.xname) 
    fgeom = TFile.Open(GeoConf.xpath)
    tegse = fgeom.Get(GeoConf.xname)

    ## wrapper switched from TEveGeoShapeExtract to TEveGeoShape by the recursive sub-import
    ## with colors and transparency being transferred 

    tegs = TEveGeoShape.ImportShapeExtract(tegse, 0)
    fgeom.Close()

    ##
    ##  Extended efforts to get visibility attributes to translate from TGeoNode thru to TEve
    ##  failed ... so set the visibilites here instead
    ##
    tegs.SetRnrSelfChildren(kFALSE,kTRUE)
    for legs in tegs.rselect(lambda self:not(self.HasChildren())):legs.SetRnrSelf(kTRUE)
    for skip in tegs.rtagged( vmr , 'SKIP' ):skip.SetRnrSelf(kFALSE)    

 
    gEve.AddGlobalElement(tegs)
    gEve.Redraw3D(True)




