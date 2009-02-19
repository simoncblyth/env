//
// namespace class to start tidy up/structuring 
// ... the idea being to factor all non-expt specifics into here 
//
// what is the state of an event display ... (mile high view )
//      .. event id
//

class EvD
{
public:
   void EvD(){
       if(gEve == NULL){ 
           TEveManager::Create();
           FitToScreen();
       }
   }

   void FitToScreen()
   {
       // Adapt the main frame to the screen size...
       Int_t qq; 
       UInt_t ww, hh;
       gVirtualX->GetWindowSize(gVirtualX->GetDefaultRootWindow(), qq, qq, ww, hh);
       Float_t screen_ratio = (Float_t)ww/(Float_t)hh;
       if (screen_ratio > 1.5) {
           gEve->GetBrowser()->MoveResize(100, 50, ww - 300, hh - 100);
           //gEve->GetBrowser()->SetWMPosition(100, 50);
       } else {
           gEve->GetBrowser()->Move(50, 50);
           //gEve->GetBrowser()->SetWMPosition(50, 50);
       }
   }


   void import_projection_geometry()
   {
       if (gRPhiMgr) {
          TEveProjectionAxes* a = new TEveProjectionAxes(gRPhiMgr);
          a->SetNdivisions(3);
          const TString rpp = "R-Phi Projection" ;
          gEve->GetScenes()->FindChild(rpp)->AddElement(a);
          gRPhiMgr->ImportElements(gGeoShape);
       }
       if (gRhoZMgr) {
          TEveProjectionAxes* a = new TEveProjectionAxes(gRhoZMgr);
          a->SetNdivisions(3);
          const TString rzp = "Rho-Z Projection" ;
          gEve->GetScenes()->FindChild(rzp)->AddElement(a);
          gRhoZMgr->ImportElements(gGeoShape);
       }
   }


   void update_projections()
   {
       // cleanup then import geometry and event 
       // in the projection managers
   
       TEveElement* top = gEve->GetCurrentEvent();
       if (gRPhiMgr && top) {
          gRPhiMgr->DestroyElements();
          gRPhiMgr->ImportElements(gGeoShape);
          gRPhiMgr->ImportElements(top);
       }
       if (gRhoZMgr && top) {
          gRhoZMgr->DestroyElements();
          gRhoZMgr->ImportElements(gGeoShape);
          gRhoZMgr->ImportElements(top);
       }
   }

   void make_gui()
   {
      // Create minimal GUI for event navigation.

      //gROOT->ProcessLine(".L SplitGLView.C+");
      //gROOT->ProcessLine(".L SplitGLView.C");
      gSystem->Load("$ENV_HOME/eve/InstallArea/$CMTCONFIG/lib/libSplitGLView.so");
   

      TEveBrowser* browser = gEve->GetBrowser();
      browser->ExecPlugin("SplitGLView", 0, "new SplitGLView(gClient->GetRoot(), 600, 450, kTRUE)");

      browser->StartEmbedding(TRootBrowser::kLeft);

      TGMainFrame* frmMain = new TGMainFrame(gClient->GetRoot(), 1000, 600);
      frmMain->SetWindowName("XX GUI");
      frmMain->SetCleanup(kDeepCleanup);

      TGHorizontalFrame* hf = new TGHorizontalFrame(frmMain);
      {
      
          TString icondir( Form("%s/icons/", gSystem->Getenv("ROOTSYS")) );
          TGPictureButton* b = 0;
          EvNav    *fh = EvNav::GetGlobal() ;

          b = new TGPictureButton(hf, gClient->GetPicture(icondir + "GoBack.gif"));
          hf->AddFrame(b, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 10, 2, 10, 10));
          b->Connect("Clicked()", "EvNav", fh, "Bck()");

          b = new TGPictureButton(hf, gClient->GetPicture(icondir + "GoForward.gif"));
          hf->AddFrame(b, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 10, 10, 10));
          b->Connect("Clicked()", "EvNav", fh, "Fwd()");

          gTextEntry = new TGTextEntry(hf);
          gTextEntry->SetEnabled(kFALSE);
          hf->AddFrame(gTextEntry, new TGLayoutHints(kLHintsLeft | kLHintsCenterY  | 
                       kLHintsExpandX, 2, 10, 10, 10));
       }
       frmMain->AddFrame(hf, new TGLayoutHints(kLHintsTop | kLHintsExpandX,0,0,20,0));

       gProgress = new TGHProgressBar(frmMain, TGProgressBar::kFancy, 100);
       gProgress->ShowPosition(kTRUE, kFALSE, "%.0f tracks");
       gProgress->SetBarColor("green");
       frmMain->AddFrame(gProgress, new TGLayoutHints(kLHintsExpandX, 10, 10, 5, 5));

       frmMain->MapSubwindows();
       frmMain->Resize();
       frmMain->MapWindow();

       browser->StopEmbedding();
       browser->SetTabTitle("Event Control", 0);
    }

};



