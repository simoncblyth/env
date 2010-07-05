import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
import os

from ROOT import kTRUE, kFALSE


class EvTrk:
    def __init__(self):
        l = ROOT.TEveLine()
        l.SetLineStyle(2)
        l.SetLineWidth(3)
        l.SetMainColor(ROOT.kRed)
        self.line = l
        ROOT.gEve.AddElement(self.line)

    def update(self, xyzs ):
        l = self.line
        for xyz in xyzs:   
            l.SetNextPoint( *xyz )
 
    def __repr__(self):
        return "<EvTrk >" 


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    trk = EvTrk()
    trk.update( [[1,1,1],[2,2,2],[3,3,3]])
    ROOT.gEve.Redraw3D(kTRUE)


