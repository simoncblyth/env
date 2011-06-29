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
      pack1 = slot.MakePack()
      pack1.SetElementName("Multi View")
      pack1.SetHorizontal()
      pack1.SetShowTitleBar(kFALSE)
      pack1.NewSlot().MakeCurrent()
      self.fXZView = gEve.SpawnNewViewer("X-Z View", "")
      self.fXZView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraOrthoXOZ)
      self.fXZView.AddScene(gEve.GetGlobalScene())
      self.fXZView.AddScene(gEve.GetEventScene())


      pack2 = pack1.NewSlot().MakePack()
      pack2.SetShowTitleBar(kFALSE)
      pack2.NewSlot().MakeCurrent()
      self.f3DView = gEve.SpawnNewViewer("3D View", "")
      self.f3DView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraPerspXOY)
      self.f3DView.AddScene(gEve.GetGlobalScene())
      self.f3DView.AddScene(gEve.GetEventScene())

      cen = [0,0,0]
      pack3 = pack1.NewSlot().MakePack()
      pack3.SetShowTitleBar(kFALSE)
      pack3.NewSlot().MakeCurrent()
      self.fYZView = gEve.SpawnNewViewer("Y-Z View", "")
      self.fYZView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraOrthoXOZ)
      self.fYZView.GetGLViewer().CurrentCamera().SetEnableRotate(kTRUE)
      self.fYZView.AddScene(gEve.GetGlobalScene())
      self.fYZView.AddScene(gEve.GetEventScene())
      
      #The last "SpawnNewViewer" has problems such that nothing is displayed
      #So to get around it, I spawn a useless new viewer and then delete it
      #Menu bar needs to be deleted first otherwise a spew of segmentation error appears ...
      fDebug = gEve.SpawnNewViewer("Debug", "")
      fDebug.GetGLViewer().DeleteMenuBar()
      fDebug.DestroyWindowAndSlot()

   # ---------------------------------------------------------------------------


if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    self.MultiView()

    gEve.Redraw3D(kFALSE, kTRUE)
