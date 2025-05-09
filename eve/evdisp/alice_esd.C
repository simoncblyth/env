/*
  alice_esd.C - a simple event-display for ALICE ESD tracks and clusters

  ------------------------------------------------------------------------
  ------------------------------------------------------------------------

  Only standard ROOT is used to process the ALICE ESD files.

  No ALICE code is needed, only four simple coordinate-transformation
  functions declared in this macro.

  A simple geometry of 10KB, extracted from the full TGeo-geometry, is
  used to outline the central detectors of ALICE.

  All files are access from the web by using the "CACHEREAD" option.

  ------------------------------------------------------------------------
  ------------------------------------------------------------------------

  1. Automatic building of ALICE ESD class declarations and dictionaries.
  ------------------------------------------------------------------------

  ALICE ESD is a TTree containing tracks and other event-related
  information with one entry per event. All these classes are part of
  the AliROOT offline framework and are not available to standard
  ROOT.

  To be able to access the event data in a natural way, by using
  data-members of classes and object containers, the header files and
  class dictionaries are automatically generated from the
  TStreamerInfo classes stored in the ESD file by using the
  TFile::MakeProject() function. The header files and a shared library
  is created in the aliesd/ directory and can be loaded dynamically
  into the ROOT session.

  See alice_esd_loadlib().
  

  2. Creation of simple GUI for event navigation.
  ------------------------------------------------------------------------

  Most common use of the event-display is to browse through a
  collection of events. Thus a simple GUI allowing this is created in
  the function make_gui().

  Eve uses the configurable ROOT-browser as its main window and so we
  create an extra tab in the left working area of the browser and
  provide backward/forward buttons.


  3. Event-navigation functions.
  ------------------------------------------------------------------------

  As this is a simple macro, we store the information about the
  current event in the global variable 'Int_t esd_event_id'. The
  functions for event-navigation simply modify this variable and call
  the load_event() function which does the following:
  1. drop the old visualization objects;
  2. retrieve given event from the ESD tree;
  3. call alice_esd_read() function to create visualization objects
     for the new event.


  4. Reading of ALICE data and creation of visualization objects.
  ------------------------------------------------------------------------

  This is performed in alice_esd_read() function, with the following
  steps:
  1. create the track container object - TEveTrackList;
  2. iterate over the ESD tracks, create TEveTrack objects and append
     them to the container;
  3. instruct the container to extrapolate the tracks and set their
     visual attributes.

*/



// Forward declarations.

class AliESDEvent;
class AliESDfriend;
class AliESDtrack;
class AliExternalTrackParam;

Bool_t     alice_esd_loadlib(const char* file, const char* project);
void       make_gui();
void       load_event();

void       alice_esd_read();
TEveTrack* esd_make_track(TEveTrackPropagator* trkProp, Int_t index, AliESDtrack* at,
			  AliExternalTrackParam* tp=0);
Bool_t     trackIsOn(AliESDtrack* t, Int_t mask);
void       trackGetPos(AliExternalTrackParam* tp, Double_t r[3]);
void       trackGetMomentum(AliExternalTrackParam* tp, Double_t p[3]);
Double_t   trackGetP(AliExternalTrackParam* tp);


// Configuration and global variables.

const char* esd_file_name         = "http://root.cern.ch/files/alice_ESDs.root";
//const char* esd_friends_file_name = "http://root.cern.ch/files/alice_ESDfriends.root";
const char* esd_friends_file_name = "http://root.cern.ch/files/AliESDfriends.root";
const char* esd_geom_file_name    = "http://root.cern.ch/files/alice_ESDgeometry.root";

TFile *esd_file          = 0;
TFile *esd_friends_file  = 0;

TTree *esd_tree          = 0;

AliESDEvent  *esd        = 0;
AliESDfriend *esd_friend = 0;

Int_t esd_event_id       = 0; // Current event id.

TEveTrackList *track_list = 0;


/******************************************************************************/
// Initialization and steering functions
/******************************************************************************/

//______________________________________________________________________________
void alice_esd()
{
   // Main function, initializes the application.
   //
   // 1. Load the auto-generated library holding ESD classes and ESD dictionaries.
   // 2. Open ESD data-files.
   // 3. Load cartoon geometry.
   // 4. Spawn simple GUI.
   // 5. Load first event.

   TFile::SetCacheFileDir(".");

   if (!alice_esd_loadlib(esd_file_name, "aliesd"))
   {
      Error("alice_esd", "Can not load project libraries.");
      return;
   }

   printf("*** Opening ESD ***\n");
   esd_file = TFile::Open(esd_file_name, "CACHEREAD");
   if (!esd_file)
      return;

   printf("*** Opening ESD-friends ***\n");
   esd_friends_file = TFile::Open(esd_friends_file_name, "CACHEREAD");
   if (!esd_friends_file)
      return;

   esd_tree = (TTree*) esd_file->Get("esdTree");
   esd_tree->GetBranch("ESDfriend.")->SetFile(esd_friends_file);
   
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

   TEveManager::Create();

   { // Simple geometry
      TFile* geom = TFile::Open(esd_geom_file_name, "CACHEREAD");
      if (!geom)
         return;
      TEveGeoShapeExtract* gse = (TEveGeoShapeExtract*) geom->Get("Gentle");
      TEveGeoShape* gsre = TEveGeoShape::ImportShapeExtract(gse, 0);
      geom->Close();
      delete geom;
      gEve->AddGlobalElement(gsre);
   }

   make_gui();

   load_event();

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

   if (track_list)
      track_list->DestroyElements();

   esd_tree->GetEntry(esd_event_id);
//   esd_tree->Show();

   alice_esd_read();

   gEve->Redraw3D(kFALSE, kTRUE);
}


/******************************************************************************/
// GUI
/******************************************************************************/

//______________________________________________________________________________
// 
// EvNavHandler class is needed to connect GUI signals.

class EvNavHandler
{
public:
   void Fwd()
   {
      if (esd_event_id < esd_tree->GetEntries() - 1) {
         ++esd_event_id;
         load_event();
      } else {
         printf("Already at last event.\n");
      }
   }
   void Bck()
   {
      if (esd_event_id > 0) {
         --esd_event_id;
         load_event();
      } else {
         printf("Already at first event.\n");
      }
   }
};

//______________________________________________________________________________
void make_gui()
{
   // Create minimal GUI for event navigation.

   TEveBrowser* browser = gEve->GetBrowser();
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
      hf->AddFrame(b);
      b->Connect("Clicked()", "EvNavHandler", fh, "Bck()");

      b = new TGPictureButton(hf, gClient->GetPicture(icondir + "GoForward.gif"));
      hf->AddFrame(b);
      b->Connect("Clicked()", "EvNavHandler", fh, "Fwd()");
   }
   frmMain->AddFrame(hf);

   frmMain->MapSubwindows();
   frmMain->Resize();
   frmMain->MapWindow();

   browser->StopEmbedding();
   browser->SetTabTitle("Event Control", 0);
}


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
    AliESDfriend *frnd   = (AliESDfriend*) esd->fESDObjects->FindObject("AliESDfriend");
    printf("Friend %p, n_tracks:%d\n", frnd, frnd->fTracks.GetEntries());

   if (track_list == 0) {
      track_list = new TEveTrackList("ESD Tracks"); 
      track_list->SetMainColor(6);
      track_list->SetMarkerColor(kYellow);
      track_list->SetMarkerStyle(4);
      track_list->SetMarkerSize(0.5);

      gEve->AddElement(track_list);
   }

   TEveTrackPropagator* trkProp = track_list->GetPropagator();
   trkProp->SetMagField( 0.1 * esdrun->fMagneticField ); // kGaus to Tesla

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
