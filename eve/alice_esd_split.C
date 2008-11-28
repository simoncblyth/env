/*
   This is based on 
       $ROOTSYS/tutorials/eve/alice_esd_split.C
   see the introductory text there

*/


// the globals that the Evd namespace class herds ...
R__EXTERN TEveProjectionManager *gRPhiMgr;
R__EXTERN TEveProjectionManager *gRhoZMgr;
TEveGeoShape *gGeoShape;
TGTextEntry *gTextEntry;
TGHProgressBar *gProgress;





// Forward declarations.

class AliESDEvent;
class AliESDfriend;
class AliESDtrack;
class AliExternalTrackParam;

Bool_t     alice_esd_loadlib(const char* file, const char* project);
void       load_event();

void       alice_esd_read();
TEveTrack* esd_make_track(TEveTrackPropagator* trkProp, Int_t index, AliESDtrack* at,
			  AliExternalTrackParam* tp=0);
Bool_t     trackIsOn(AliESDtrack* t, Int_t mask);
void       trackGetPos(AliExternalTrackParam* tp, Double_t r[3]);
void       trackGetMomentum(AliExternalTrackParam* tp, Double_t p[3]);
Double_t   trackGetP(AliExternalTrackParam* tp);


// Configuration and global variables.



TTree *esd_tree          = 0;

AliESDEvent  *esd        = 0;
AliESDfriend *esd_friend = 0;

Int_t esd_event_id       = 0; // Current event id.

TEveTrackList *track_list = 0;

/******************************************************************************/
// Initialization and steering functions
/******************************************************************************/

//______________________________________________________________________________
void alice_esd_split()
{
   // Main function, initializes the application.
   //
   // 1. Load the auto-generated library holding ESD classes and ESD dictionaries.
   // 2. Open ESD data-files.
   // 3. Load cartoon geometry.
   // 4. Spawn simple GUI.
   // 5. Load first event.

   TFile::SetCacheFileDir(".");

   const char* esd_file_name         = "http://root.cern.ch/files/alice_ESDs.root";
   if (!alice_esd_loadlib(esd_file_name, "aliesd"))
   {
      Error("alice_esd", "Can not load project libraries.");
      return;
   }

   printf("*** Opening ESD ***\n");
   TFile *esd_file          = 0;
   esd_file = TFile::Open(esd_file_name, "CACHEREAD");
   if (!esd_file)
      return;

   printf("*** Opening ESD-friends ***\n");
   const char* esd_friends_file_name = "http://root.cern.ch/files/alice_ESDfriends.root";
   TFile *esd_friends_file  = 0;
   esd_friends_file = TFile::Open(esd_friends_file_name, "CACHEREAD");
   if (!esd_friends_file)
      return;

   esd_tree = (TTree*) esd_file->Get("esdTree");

   esd = (AliESDEvent*) esd_tree->GetUserInfo()->FindObject("AliESDEvent");

   // Set the branch addresses.
   {
      TIter next(esd->fESDObjects);
      TObject *el;
      while ((el=(TNamed*)next()))
      {
         TString bname(el->GetName());
         if(bname.CompareTo("AliESDfriend")==0)
         {
            // AliESDfriend needs some '.' magick.
            esd_tree->SetBranchAddress("ESDfriend.", esd->fESDObjects->GetObjectRef(el));
         }
         else
         {
            esd_tree->SetBranchAddress(bname, esd->fESDObjects->GetObjectRef(el));
         }
      }
   }


   EvDisp::EvDisp();
   gROOT->ProcessLine(".x evd_geometry.C");
   EvDisp::make_gui();
   EvDisp::import_projection_geometry();

   load_event();

   EvDisp::update_projections();
   gEve->Redraw3D(kTRUE); // Reset camera after the first event has been shown.
}

//______________________________________________________________________________
Bool_t alice_esd_loadlib(const char* file, const char* project)
{
   // Make sure that shared library created from the auto-generated project
   // files exists and load it.

   TString lib(Form("%s/%s.%s", project, project, gSystem->GetSoExt()));

   if (gSystem->AccessPathName(lib, kReadPermission)) {
      TFile* f = TFile::Open(file, "CACHEREAD");
      if (f == 0)
         return kFALSE;
      f->MakeProject(project, "*", "++");
      f->Close();
      delete f;
   }
   return gSystem->Load(lib) >= 0;
}

//______________________________________________________________________________
void load_event()
{
   // Load event specified in global esd_event_id.
   // The contents of previous event are removed.

   printf("Loading event %d.\n", esd_event_id);
   gTextEntry->SetTextColor(0xff0000);
   gTextEntry->SetText(Form("Loading event %d...",esd_event_id));
   gSystem->ProcessEvents();

   if (track_list)
      track_list->DestroyElements();

   esd_tree->GetEntry(esd_event_id);

   alice_esd_read();  // updates the track_list

   gEve->Redraw3D(kFALSE, kTRUE);
   gTextEntry->SetTextColor(0x000000);
   gTextEntry->SetText(Form("Event %d loaded",esd_event_id));
   gROOT->ProcessLine("SplitGLView::UpdateSummary()");
}

//______________________________________________________________________________




//
// namespace class to start tidy up/structuring 
// ... the idea being to factor all non-expt specifics into here 
//
// what is the state of an event display ... (mile high view )
//      .. event id
//

class EvDisp
{
public:
   void EvDisp(){
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
          EvNavHandler    *fh = new EvNavHandler;

          b = new TGPictureButton(hf, gClient->GetPicture(icondir + "GoBack.gif"));
          hf->AddFrame(b, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 10, 2, 10, 10));
          b->Connect("Clicked()", "EvNavHandler", fh, "Bck()");

          b = new TGPictureButton(hf, gClient->GetPicture(icondir + "GoForward.gif"));
          hf->AddFrame(b, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 10, 10, 10));
          b->Connect("Clicked()", "EvNavHandler", fh, "Fwd()");

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



class EvNavHandler
{
public:
   void Fwd()
   {
      if (esd_event_id < esd_tree->GetEntries() - 1) {
         ++esd_event_id;
         load_event();
         Evd::update_projections();
      } else {
         gTextEntry->SetTextColor(0xff0000);
         gTextEntry->SetText("Already at last event");
         printf("Already at last event.\n");
      }
   }
   void Bck()
   {
      if (esd_event_id > 0) {
         --esd_event_id;
         load_event();
         Evd::update_projections();
      } else {
         gTextEntry->SetTextColor(0xff0000);
         gTextEntry->SetText("Already at first event");
         printf("Already at first event.\n");
      }
   }
};


/******************************************************************************/
// Code for reading AliESD and creating visualization objects
/******************************************************************************/

enum ESDTrackFlags {
   kITSin=0x0001,kITSout=0x0002,kITSrefit=0x0004,kITSpid=0x0008,
   kTPCin=0x0010,kTPCout=0x0020,kTPCrefit=0x0040,kTPCpid=0x0080,
   kTRDin=0x0100,kTRDout=0x0200,kTRDrefit=0x0400,kTRDpid=0x0800,
   kTOFin=0x1000,kTOFout=0x2000,kTOFrefit=0x4000,kTOFpid=0x8000,
   kHMPIDpid=0x20000,
   kEMCALmatch=0x40000,
   kTRDbackup=0x80000,
   kTRDStop=0x20000000,
   kESDpid=0x40000000,
   kTIME=0x80000000
};

//______________________________________________________________________________
void alice_esd_read()
{
   // Read tracks and associated clusters from current event.

   AliESDRun    *esdrun = (AliESDRun*)    esd->fESDObjects->FindObject("AliESDRun");
   TClonesArray *tracks = (TClonesArray*) esd->fESDObjects->FindObject("Tracks");

   // This needs further investigation. Clusters not shown.
   // AliESDfriend *frnd   = (AliESDfriend*) esd->fESDObjects->FindObject("AliESDfriend");
   // printf("Friend %p, n_tracks:%d\n", frnd, frnd->fTracks.GetEntries());

   if (track_list == 0) {
      track_list = new TEveTrackList("ESD Tracks"); 
      track_list->SetMainColor(6);
      //track_list->SetLineWidth(2);
      track_list->SetMarkerColor(kYellow);
      track_list->SetMarkerStyle(4);
      track_list->SetMarkerSize(0.5);

      gEve->AddElement(track_list);
   }

   TEveTrackPropagator* trkProp = track_list->GetPropagator();
   trkProp->SetMagField( 0.1 * esdrun->fMagneticField ); // kGaus to Tesla

   gProgress->Reset();
   gProgress->SetMax(tracks->GetEntriesFast());
   for (Int_t n=0; n<tracks->GetEntriesFast(); ++n)
   {
      AliESDtrack* at = (AliESDtrack*) tracks->At(n);

      // If ITS refit failed, take track parameters at inner TPC radius.
      AliExternalTrackParam* tp = at;
      if (! trackIsOn(at, kITSrefit)) {
         tp = at->fIp;
      }

      TEveTrack* track = esd_make_track(trkProp, n, at, tp);
      track->SetAttLineAttMarker(track_list);
      gEve->AddElement(track, track_list);

      // This needs further investigation. Clusters not shown.
      // if (frnd)
      // {
      //     AliESDfriendTrack* ft = (AliESDfriendTrack*) frnd->fTracks->At(n);
      //     printf("%d friend = %p\n", ft);
      // }
      gProgress->Increment(1);
   }

   track_list->MakeTracks();
}

//______________________________________________________________________________
TEveTrack* esd_make_track(TEveTrackPropagator*   trkProp,
			  Int_t                  index,
			  AliESDtrack*           at,
			  AliExternalTrackParam* tp)
{
   // Helper function creating TEveTrack from AliESDtrack.
   //
   // Optionally specific track-parameters (e.g. at TPC entry point)
   // can be specified via the tp argument.

   Double_t      pbuf[3], vbuf[3];
   TEveRecTrack  rt;

   if (tp == 0) tp = at;

   rt.fLabel  = at->fLabel;
   rt.fIndex  = index;
   rt.fStatus = (Int_t) at->fFlags;
   rt.fSign   = (tp->fP[4] > 0) ? 1 : -1;

   trackGetPos(tp, vbuf);      rt.fV.Set(vbuf);
   trackGetMomentum(tp, pbuf); rt.fP.Set(pbuf);

   Double_t ep = trackGetP(at);
   Double_t mc = 0.138; // at->GetMass(); - Complicated funciton, requiring PID.

   rt.fBeta = ep/TMath::Sqrt(ep*ep + mc*mc);
 
   TEveTrack* track = new TEveTrack(&rt, trkProp);
   track->SetName(Form("TEveTrack %d", rt.fIndex));
   track->SetStdTitle();

   return track;
}

//______________________________________________________________________________
Bool_t trackIsOn(AliESDtrack* t, Int_t mask)
{
   // Check is track-flag specified by mask are set.

   return (t->fFlags & mask) > 0;
}

//______________________________________________________________________________
void trackGetPos(AliExternalTrackParam* tp, Double_t r[3])
{
   // Get global position of starting point of tp.

  r[0] = tp->fX; r[1] = tp->fP[0]; r[2] = tp->fP[1];

  Double_t cs=TMath::Cos(tp->fAlpha), sn=TMath::Sin(tp->fAlpha), x=r[0];
  r[0] = x*cs - r[1]*sn; r[1] = x*sn + r[1]*cs;
}

//______________________________________________________________________________
void trackGetMomentum(AliExternalTrackParam* tp, Double_t p[3])
{
   // Return global momentum vector of starting point of tp.

   p[0] = tp->fP[4]; p[1] = tp->fP[2]; p[2] = tp->fP[3];

   Double_t pt=1./TMath::Abs(p[0]);
   Double_t cs=TMath::Cos(tp->fAlpha), sn=TMath::Sin(tp->fAlpha);
   Double_t r=TMath::Sqrt(1 - p[1]*p[1]);
   p[0]=pt*(r*cs - p[1]*sn); p[1]=pt*(p[1]*cs + r*sn); p[2]=pt*p[2];
}

//______________________________________________________________________________
Double_t trackGetP(AliExternalTrackParam* tp)
{
   // Return magnitude of momentum of tp.

   return TMath::Sqrt(1.+ tp->fP[3]*tp->fP[3])/TMath::Abs(tp->fP[4]);
}
