import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
import os
import math 
from ROOT import kTRUE, kFALSE

#counter = 0

class EvTrk(list):
    def __init__(self):
        pass
    def _line(self):
        l = ROOT.TEveLine()
        l.SetLineStyle(1)
	l.SetSmooth(1)
        l.SetLineWidth(3)
        l.SetMainColor(ROOT.kWhite)
        return l

    def clear(self):
	for i in range(len(self)):
	    k = self.pop()
            k.Destroy()
        
    def update(self, allft, xysc=0.1 ):

	ntrk = allft.GetNTrack()
	zs = [-150,0,150]

	count = 0
	while (count < ntrk):
	    ft = allft.Get(count)
	    l = self._line()
            for z in zs:    
                l.SetNextPoint(ft.X().At((z+118.7)*10)*xysc,ft.Y().At((z+118.7)*10)*xysc, z )
        
	    if (count == 0):
		l.SetMainColor(ROOT.kRed)
		l.SetLineStyle(9)
	    else:
		l.SetMainTransparency(95)
	    
	    #name = "Track #%i \nTheta: %s +/- %s" % (count,ft.Theta(),ft.ThetaError()*100)

	    name = "Track #%i \nTheta: %s +/- %s\nPhi: %s +/- %s\nFitness: %s\nChiSquare: %s\nNFitLayer: %i" % (count,ft.Theta(),ft.ThetaError(),ft.Phi(),ft.PhiError(),ft.GetFitness(),ft.GetChisquare(),ft.GetNFitLayer())
	    #chi = ft.GetChisquare()
	    #transp = int(100/ntrk*count)
	    #print transp
	    #l.SetMainTransparency(transp)
	    #name = ft.Print()
	    #type(name)
	    l.SetElementTitle(name)    
	    self.append(l)
            ROOT.gEve.AddElement(l)
	    count = count +1



if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    trk = EvTrk()
    trk.update( [[1,1,1],[2,2,2],[3,3,3]])
    ROOT.gEve.Redraw3D(kTRUE)

