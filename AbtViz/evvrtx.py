import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
import os
import math

from ROOT import kTRUE, kFALSE


class EvVrtx(list):
    def __init__(self):
        pass
    def _marker(self):
        p = ROOT.TEvePointSet()
        p.SetMarkerStyle(4)
	p.SetMainColor(ROOT.kYellow)
        return p

    def clear(self):
	for i in range(len(self)):
	    k = self.pop()
	    k.Destroy()   

    def update(self, vrtxp , xysc=0.1 ):
        p = self._marker()
 	p.SetMarkerSize((vrtxp[3]/5000))
        p.SetNextPoint(vrtxp[0]*xysc,vrtxp[1]*xysc,(vrtxp[2]-295.0)*xysc)
	
	tag = "Vertex: \n (x = %d, y = %d, z = %d) mm \n Number of Photons = %d" % (vrtxp[0],vrtxp[1],vrtxp[2],vrtxp[3])
	p.SetElementTitle(tag)	

        self.append(p)
        ROOT.gEve.AddElement(p)



if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    vrtx = EvVtrx()
    vrtx.update( [[1,1,1],[2,2,2],[3,3,3]])
    ROOT.gEve.Redraw3D(kTRUE)


