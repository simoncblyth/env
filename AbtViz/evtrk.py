import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
import os

from ROOT import kTRUE, kFALSE


class EvTrk(list):
    def __init__(self):
        pass
    def _line(self):
        l = ROOT.TEveLine()
        l.SetLineStyle(2)
        l.SetLineWidth(3)
        l.SetMainColor(ROOT.kRed)
        return l

    def update(self, xyzs , reset=True, xysc=0.1 ):
        if reset:
            for i in range(len(self)):
                k = self.pop()
                k.Destroy()
        l = self._line()
        for x,y,z in xyzs:   
            l.SetNextPoint(x*xysc,y*xysc, z )
        self.append(l)
        ROOT.gEve.AddElement(l)



if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    trk = EvTrk()
    trk.update( [[1,1,1],[2,2,2],[3,3,3]])
    ROOT.gEve.Redraw3D(kTRUE)


