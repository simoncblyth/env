"""
   translated from $ROOTSYS/tutorials/eve/lineset_test.C
   see
      https://savannah.cern.ch/bugs/?40942
  
   Fixed by PyRoot change circa 2008-09-24 19:04

"""
import sys
import ROOT
if hasattr(ROOT.PyConfig,'GUIThreadScheduleOnce'):
    ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
else:
    print "you need to get a more recent ROOT to run this "
    sys.exit(1)

def lineset_test(nlines = 40, nmarkers = 4):
    r = ROOT.TRandom(0)
    s = 100

    ls = ROOT.TEveStraightLineSet()

    for i in range(nlines):
        ls.AddLine( r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s) ,
                    r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s))
        nm = int(nmarkers*r.Rndm())
        for m in range(nm):
            ls.AddMarker( i, r.Rndm() )
    ls.SetMarkerSize(1.5)
    ls.SetMarkerStyle(4)

    ROOT.gEve.AddElement(ls)
    ROOT.gEve.Redraw3D()
    return ls

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    lineset_test()
