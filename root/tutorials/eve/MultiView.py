import ROOT
ROOT.PyConfig.GUIThreadScheduleOnce += [ ROOT.TEveManager.Create ]
import os

'''
// Multi-view (3d, rphi, rhoz) service class using EVE Window Manager.
// Author: Matevz Tadel 2009

// MultiView
//
// Structure encapsulating standard views: 3D, r-phi and rho-z.
// Includes scenes and projection managers.
//
// Should be used in compiled mode.
'''

class MultiView(list):

   def __init__(self):
      from ROOT import kTRUE, kFALSE, kWhite, gEve	
      # Constructor --- creates required scenes, projection managers
      # and GL viewers.

      # Scenes
      #========
	
      self.fRPhiGeomScene  = gEve.SpawnNewScene("RPhi Geometry",
                                            "Scene holding projected geometry for the RPhi view.")
      self.fRhoZGeomScene  = gEve.SpawnNewScene("RhoZ Geometry",
                                            "Scene holding projected geometry for the RhoZ view.")
      self.fRPhiEventScene = gEve.SpawnNewScene("RPhi Event Data",
                                            "Scene holding projected event-data for the RPhi view.")
      self.fRhoZEventScene = gEve.SpawnNewScene("RhoZ Event Data",
                                            "Scene holding projected event-data for the RhoZ view.")


      # Projection managers
      # =====================

      self.fRPhiMgr = ROOT.TEveProjectionManager(ROOT.TEveProjection.kPT_RPhi)
      gEve.AddToListTree(self.fRPhiMgr, kFALSE)
     
       
      self.a = ROOT.TEveProjectionAxes(self.fRPhiMgr)
      self.a.SetMainColor(kWhite)
      self.a.SetTitle("R-Phi")
      self.a.SetTitleSize(0.05)
      self.a.SetTitleFont(102)
      self.a.SetLabelSize(0.025)
      self.a.SetLabelFont(102)
      self.fRPhiGeomScene.AddElement(self.a)
            

      self.fRhoZMgr = ROOT.TEveProjectionManager(ROOT.TEveProjection.kPT_RhoZ)
      gEve.AddToListTree(self.fRhoZMgr, kFALSE)
      
      self.a = ROOT.TEveProjectionAxes(self.fRhoZMgr)
      self.a.SetMainColor(kWhite)
      self.a.SetTitle("Rho-Z")
      self.a.SetTitleSize(0.05)
      self.a.SetTitleFont(102)
      self.a.SetLabelSize(0.025)
      self.a.SetLabelFont(102)
      self.fRhoZGeomScene.AddElement(self.a)
      


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
      self.f3DView.AddScene(gEve.GetGlobalScene())
      self.f3DView.AddScene(gEve.GetEventScene())

      pack = pack.NewSlot().MakePack()
      pack.SetShowTitleBar(kFALSE)
      
      pack.NewSlot().MakeCurrent()
      self.fRPhiView = gEve.SpawnNewViewer("RPhi View", "")
      #self.fRPhiView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraOrthoXOY)
      self.fRPhiView.AddScene(self.fRPhiGeomScene)
      self.fRPhiView.AddScene(self.fRPhiEventScene)


      pack.NewSlot().MakeCurrent()
      self.fRhoZView = gEve.SpawnNewViewer("RhoZ View", "")
      #self.fRhoZView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraOrthoXOY)
      self.fRhoZView.AddScene(self.fRhoZGeomScene)
      self.fRhoZView.AddScene(self.fRhoZEventScene)
      
      pack.NewSlot().MakeCurrent()
      self.fRhoZView = gEve.SpawnNewViewer("RhoZ View", "")
      #self.fRhoZView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraOrthoXOY)
      self.fRhoZView.AddScene(self.fRhoZGeomScene)
      self.fRhoZView.AddScene(self.fRhoZEventScene)

      pack.NewSlot().MakeCurrent()
      self.fRhoZView = gEve.SpawnNewViewer("RhoZ View", "")
      #self.fRhoZView.GetGLViewer().SetCurrentCamera(ROOT.TGLViewer.kCameraOrthoXOY)
      self.fRhoZView.AddScene(self.fRhoZGeomScene)
      self.fRhoZView.AddScene(self.fRhoZEventScene)
			

   # ---------------------------------------------------------------------------

   def SetDepth(self, d):
   
      # Set current depth on all projection managers.

      self.fRPhiMgr.SetCurrentDepth(d)
      self.fRhoZMgr.SetCurrentDepth(d)
   

   # ---------------------------------------------------------------------------

   def ImportGeom(self, el):
    
      self.fRPhiMgr.ImportElements(el, self.fRPhiGeomScene)
     
      self.fRhoZMgr.ImportElements(el, self.fRhoZGeomScene)
   

   def ImportEvent(self, el):
    
      self.fRPhiMgr.ImportElements(el, self.fRPhiEventScene)
       
      self.fRhoZMgr.ImportElements(el, self.fRhoZEventScene)
   

   # ---------------------------------------------------------------------------

   def DestroyEvent(self):
   
      self.fRPhiEventScene.DestroyElements()
        
      self.fRhoZEventScene.DestroyElements()

if __name__=='__main__':
    ROOT.PyGUIThread.finishSchedule()
    self.MultiView()

    gEve.Redraw3D(kFALSE, kTRUE)
