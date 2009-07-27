

from ROOT import kTRUE,kFALSE
import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]

def quad_test(x=0, y=0, z=0, num=100, register=kTRUE):
   
   from ROOT import TRandom, gStyle
   r = TRandom(0)
   ## gStyle.SetPalette(1, 0)  2nd arg difficult from python :  Int_t* colors = 0  so just leave as default
   gStyle.SetPalette(1)

   from ROOT import TEveRGBAPalette, TEveFrameBox, TEveQuadSet, kGray

   pal = TEveRGBAPalette(0, 130)
   box = TEveFrameBox()
   box.SetAAQuadXY(-10, -10, 0, 20, 20)
   box.SetFrameColor(kGray)

   q = TEveQuadSet("RectangleXY")
   q.SetOwnIds(kTRUE)
   q.SetPalette(pal)
   q.SetFrame(box)
   q.Reset(TEveQuadSet.kQT_RectangleXY, kFALSE, 32)

   from ROOT import TNamed
   for i in range(num):
       q.AddQuad(r.Uniform(-10, 9), r.Uniform(-10, 9), 0, r.Uniform(0.2, 1), r.Uniform(0.2, 1))
       v = r.Uniform(0,130)
       q.QuadValue(int(v))   ## must be int 
       q.QuadId(TNamed("QuadIdx %d" % i, "TNamed assigned to a quad as an indentifier."))
 
   q.RefitPlex()

   from ROOT import TMath
   t = q.RefMainTrans()
   t.RotateLF(1, 3, 0.5*TMath.Pi())
   t.SetPos(x, y, z)

   if register:
      from ROOT import gEve
      gEve.AddElement(q)
      gEve.Redraw3D(kTRUE)
   return q

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    q = quad_test() 
