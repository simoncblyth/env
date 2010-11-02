import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
import os

class MultiView(list):

   def __init__(self):
      from ROOT import kTRUE, kFALSE, gEve	
   

      # Viewers
      # =========

      slot = 0
      pack = 0

      slot = ROOT.TEveWindow.CreateWindowInTab(gEve.GetBrowser().GetTabRight())
      pack = slot.MakePack()
      pack.SetElementName("Multi View")
      pack.SetHorizontal()
      pack.SetShowTitleBar(kFALSE)
      pack.NewSlot().MakeCurrent()
      self.f3DView = gEve.SpawnNewViewer("3D View", "")
      self.f3DView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraPerspXOY)
      self.f3DView.AddScene(gEve.GetGlobalScene())
      self.f3DView.AddScene(gEve.GetEventScene())

      pack = pack.NewSlot().MakePack()
      pack.SetShowTitleBar(kFALSE)
      
      pack.NewSlot().MakeCurrent()
      self.fXZView = gEve.SpawnNewViewer("X-Z View", "")
      self.fXZView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraOrthoXOZ)
      self.fXZView.AddScene(gEve.GetGlobalScene())
      self.fXZView.AddScene(gEve.GetEventScene())

      pack = pack.NewSlot().MakePack()
      pack.SetShowTitleBar(kFALSE)
      pack.NewSlot().MakeCurrent()
      self.fYZView = gEve.SpawnNewViewer("Y-Z View", "")
      self.fYZView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraOrthoZOY)
      self.fYZView.AddScene(gEve.GetGlobalScene())
      self.fYZView.AddScene(gEve.GetEventScene())
      
      #A really bad way of fixing a problem I know nothing about
      #Basically the last "SpawnNewViewer" has problems such that nothing is displayed
      #So to get around it, I spawn a useless new viewer and then delete it ...
      fDebug = gEve.SpawnNewViewer("Debug", "")
      fDebug.DestroyWindowAndSlot()

   # ---------------------------------------------------------------------------


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    self.MultiView()

    gEve.Redraw3D(kFALSE, kTRUE)
