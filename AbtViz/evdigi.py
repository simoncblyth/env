import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
import os

from ROOT import kTRUE, kFALSE
import env.root.addons.TEveDigitSet_Additions



class PMTDigi:
    def __init__(self, pmt, palette):
        self.offset = 0,0,-30.     ## arbitrary offset value ... chosen by eye ... what should this be ?
        self.pmt    = pmt 
        self.index  = pmt.id
        name = repr(pmt)
        self.name = name
        self.title = name
        self.set_coords(pmt)
        self.line = self.create_lineset()
        self.cone = self.create_coneset(palette)

    def set_coords(self, pmt):
        xyz = pmt.x , pmt.y, pmt.z
        dir = ROOT.TEveVector() 
        pos = ROOT.TEveVector() 
        dir.Set(pmt.x,pmt.y,0.)
        dir *= -1./dir.Mag()
        dir *= 10.
        pos.Set(*xyz)
        pos *= 0.1
        end = pos + dir*1.3

        self.dir = dir
        self.pos = pos
        self.end = end

    def create_lineset(self):
        lineset = ROOT.TEveStraightLineSet(self.name)
        lineset.SetLineColor(ROOT.kYellow)
        lineset.SetLineWidth(2)
        lineset.SetElementTitle(self.title)   ## used as tooltip   
        pos = self.pos
        end = self.end
        lineset.AddLine(pos.fX, pos.fY, pos.fZ, end.fX, end.fY, end.fZ)
        tlines = lineset.RefMainTrans()
        tlines.SetPos(*self.offset)
        return lineset

    def create_coneset(self, palette ): 
        coneset = ROOT.TEveBoxSet(self.name)
        coneset.Reset(ROOT.TEveBoxSet.kBT_EllipticCone, EvDigi.valIsColor, 64)
        coneset.SetPickable(kTRUE)
        coneset.SetDrawConeCap(kTRUE)
        coneset.SetPalette(palette)
        coneset.SetElementTitle(self.title)   ## used as tooltip   

        ## elliptical cone, has ellipse for the base ... with majors r1, r2  
        r1 , r2, ang = 10., 10., 180.
        coneset.AddEllipticCone( self.pos, self.dir, r1, r2, ang )
        coneset.RefitPlex()

        tcones = coneset.RefMainTrans()
        tcones.SetPos(*self.offset)
        return coneset

    def update(self, resp ):   
        if EvDigi.valIsColor: 
            rgba = 0.01*resp, 0.02*resp, 0.03*resp, 1
            self.cone.SetDigitColorRGBA( i , *rgba )
        else:
            self.cone.SetDigitValue( 0 , int(resp) )   ## only one so index is zero
        self.cone.ElementChanged(update_scenes=kTRUE, redraw=kTRUE)

    def __call__(self, resp ):
        self.update(resp)

    def add_(self):
        ROOT.gEve.AddElement(self.cone)
        ROOT.gEve.AddElement(self.line)

    def __repr__(self):
        return "<PMTDigi %s >" % self.name

       
class EvDigi(list):

    valIsColor = kFALSE ## i guess this makes no sense when using palette
    
    def __init__(self, pmtmin=0 , pmtmax=150 ):
        self.pmtmin = pmtmin
        self.pmtmax = pmtmax
        ROOT.gStyle.SetPalette(1)
        palette = ROOT.TEveRGBAPalette(self.pmtmin, self.pmtmax)
        self.palette = palette

        from pmtmap import PMTMap
        pmtmap = PMTMap()
        self.setup_pmt(pmtmap)
        self.pmtmap = pmtmap
 
    def setup_pmt(self, pm ):
        for i,pmt in enumerate(pm):
            name = "PMT %d" % ( i + 1 )
            pd = PMTDigi( pmt , self.palette )       
            self.append(pd)

    def update_pmt(self , resp ):
        assert len(resp) == 16
        for i, r in enumerate(resp):
            self[i].update(int(r))  
       

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    digi = EvDigi()
    for pd in digi:
        pd.add_()


    ROOT.gEve.Redraw3D(kTRUE)


