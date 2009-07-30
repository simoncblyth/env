"""
   Translation of  tutorials/eve/elliptic_cone_test.C
  
   Only tricky aspect was the DigitColor call ...
   due to UChar_t arguments 
   worked out different way from :
        http://root.cern.ch/root/html/src/TColor.cxx.html

"""
import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

def elliptic_cone_test(x=0, y=0, z=0, num=100):
    from ROOT import kYellow, kTRUE, kFALSE
    from ROOT import TEveStraightLineSet
    lines = TEveStraightLineSet("StraightLines")
    lines.SetLineColor(kYellow)
    lines.SetLineWidth(2)

    from ROOT import TRandom
    r = TRandom()

    from ROOT import TEveBoxSet
    cones = TEveBoxSet("EllipticConeSet")
    cones.Reset(TEveBoxSet.kBT_EllipticCone, kTRUE, 64)
    cones.SetPickable(kTRUE)

    a = 40 # max distance between cones
    from ROOT import TEveVector
    dir = TEveVector() 
    pos = TEveVector() 

    from ROOT import TMath, TColor
    for i in range(num):
        theta  = r.Uniform(0,TMath.Pi())
        phi    = r.Uniform (-TMath.Pi(), TMath.Pi())
        height = r.Uniform(5, 15)
        rad    = r.Uniform(3, 5)
        dir.Set(TMath.Cos(phi)*TMath.Cos(theta), TMath.Sin(phi)*TMath.Cos(theta), TMath.Sin(theta))
        dir *= height
        pos.Set(r.Uniform(-a,a), r.Uniform(-a, a), r.Uniform(-a, a))
        cones.AddEllipticCone(pos, dir, rad, 0.5*rad, r.Uniform(0,360))

        ci = 1001 + i 
        tc = TColor( ci , r.Uniform(0.1,1), r.Uniform(0.1,1), r.Uniform(0.1, 1), "mycol%s" % i , r.Uniform(0.1, 1)) 
        #print tc
        cones.DigitColor( ci )

        # draw axis line 30% longer than cone height
        end = pos + dir*1.3
        lines.AddLine(pos.fX, pos.fY, pos.fZ, end.fX, end.fY, end.fZ)
  

    # by default cone cap not drawn
    cones.SetDrawConeCap(kTRUE)

    cones.RefitPlex()
    t = cones.RefMainTrans()
    t.SetPos(x, y, z)

    from ROOT import gEve
    gEve.AddElement(cones)
    gEve.AddElement(lines)
    gEve.Redraw3D(kTRUE)

    return cones


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    elliptic_cone_test()
