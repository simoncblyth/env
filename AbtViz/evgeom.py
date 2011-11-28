import os
import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

from ROOT import kTRUE, kFALSE

class EvGeom:
    def __init__(self):
        from geoconf import GeoConf, VolMatcher
        self.geo = self.load_geo(GeoConf)
        self.vmr = VolMatcher()
        self.setup_viz()
        self.create_hitmap()

    def load_geo(self,conf):
        """
            load the extracted geometry created by prepare_geom.py
        """
        xpath = os.path.join( os.path.dirname(__file__) , conf.xpath )
        assert os.path.exists( xpath ), "ABORT the extracted geometry file %s does not exist, create it using prepare_geom.py " % xpath
        fgeom = ROOT.TFile.Open( xpath )
        tegse = fgeom.Get( conf.xname )
        tegs = ROOT.TEveGeoShape.ImportShapeExtract(tegse, 0)
        fgeom.Close()
        return tegs

    def create_hitmap(self):
        """
            create the mapping between layer number and detector id used 
            in the data model and the volumes of the geometry 
            ... current mapping is a guess 
        """
        self.hitmap = {}
        for lay in range(7):
            self.hitmap[lay] = {}   
            for det,vol in enumerate(self.tagged("L%d" % lay )):
                 self.hitmap[lay][det] = vol  

    def clear_hits(self):
        for vol in self.selected():
            vol.SelectElement(kFALSE)

    def update_hits(self , hits ):
        """
             tracker hit presentation by selection of volumes 
        """
        for lay,det in hits:
            vol = self.hitmap[lay].get(det,None)
            if vol:vol.SelectElement(kTRUE)
        self.geo.ElementChanged(update_scenes=kTRUE, redraw=kTRUE)

    def setup_viz(self):   
        """
             setup vizibility of volumes ... failed to propagated these 
             settings thru extraction/importing, so have to redo it here 
        """ 
        import env.root.addons.TEveElement_Additions
        self.geo.SetRnrSelfChildren(kFALSE,kTRUE)

       
        for leaf in self.geo.rselect(lambda self:not(self.HasChildren())):
            leaf.SetRnrSelf(kTRUE)
	    leaf.SetMainTransparency(80)

        for skip in self.geo.rtagged( self.vmr , 'SKIP' ):
            skip.SetRnrSelf(kFALSE)    

    def selected(self):
        return self.geo.rselect(lambda v:v.GetSelectedLevel() != "\x00" )

    def tagged(self, tag ):
        return  self.geo.rtagged( self.vmr , tag )

    def __call__(self, patn, **kwa ):
        return self.geo.rmatch(patn, **kwa)





if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    eg = EvGeom()
    ROOT.gEve.AddGlobalElement(eg.geo)

    # NB the trailing $ ... otherwise would find  _20 _21 _21 ... _29
    eg.geo.rfind("1m_Plastic_Scintillator_log_2$").SelectElement(kTRUE)

    for v in eg("1m_Plastic_Scintillator_log_2"):
        v.SelectElement(kTRUE)

    eg.create_hitmap()
    
    ROOT.gEve.Redraw3D(kTRUE)
