/*
     Alice Event Reader 

     Handy one liners for debugging from cint :
         AliEvReader* aer = (AliEvReader*)EvReader::GetEvReader(EvReader::kAlice);

*/

class AliESDEvent;
class AliESDfriend;
class AliESDtrack;
class AliExternalTrackParam;

class AliEvReader : public IEvReader 
{

   public:

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


     void AliEvReader();

     // methods demanded my the interface

     void Read();        // updates the track_list
     Bool_t LoadProject(const char* file , const char* project);
     Bool_t InitProject(const char* file , const char* project);
     Bool_t LoadGeometry(const char* file , const char* shape);

     Bool_t InitProject();
     Bool_t LoadGeometry();

     // internal expt specific methods 
     //       i do not like TEve* in here ...

     TEveTrack* esd_make_track(TEveTrackPropagator* trkProp, Int_t index, AliESDtrack* at, AliExternalTrackParam* tp=0);
     Double_t trackGetP(AliExternalTrackParam* tp);
     void     trackGetMomentum(AliExternalTrackParam* tp, Double_t p[3]);
     void     trackGetPos(AliExternalTrackParam* tp, Double_t r[3]);
     Bool_t   trackIsOn(AliESDtrack* t, Int_t mask);

     // constants 

     static const char* esd_file_name ;
     static const char* esd_friends_file_name ;
     static const char* esd_project_name ;

     static const char* esd_geom_file_name ;
     static const char* esd_geom_name ;

     // state : convenience pointers 

     AliESDEvent* esd  ;
     AliESDfriend *esd_friend ;

};

const char* AliEvReader::esd_file_name         = "http://root.cern.ch/files/alice_ESDs.root";
const char* AliEvReader::esd_friends_file_name = "http://root.cern.ch/files/alice_ESDfriends.root";
const char* AliEvReader::esd_project_name      = "aliesd" ;

const char* AliEvReader::esd_geom_file_name    = "http://root.cern.ch/files/alice_ESDgeometry.root";
const char* AliEvReader::esd_geom_name         = "Gentle" ;



void AliEvReader::AliEvReader()
{
         esd = 0 ;
         esd_friend = 0 ;
}


Bool_t AliEvReader::InitProject()
{
     InitProject( esd_file_name , esd_project_name );
}

Bool_t AliEvReader::InitProject(const char* file , const char* project )
{

   printf("*** AliEvReader::InitProject file:%s project:%s \n", file, project  ); 

   TFile::SetCacheFileDir(".");
   if (!LoadProject(file , project ))
   {
      Error("alice_esd", "Can not load project libraries.");
      return kFALSE;
   }

   printf("*** AliEvReader::InitProject  Opening ESD ***\n");

   esd_file = TFile::Open(file , "CACHEREAD");
   if (!esd_file)
      return;

   printf("*** Opening ESD-friends %s ***\n", esd_friends_file_name );
   esd_friends_file = TFile::Open(esd_friends_file_name, "CACHEREAD");
   if (!esd_friends_file)
      return;

   esd_tree = (TTree*) esd_file->Get("esdTree");
   if(!esd_tree){
       Error("alice_esd", "cannot access tree " );
   } else {
       printf("*** accessed tree *** \n");  
   }


   esd = (AliESDEvent*) esd_tree->GetUserInfo()->FindObject("AliESDEvent");
   if(!esd){
       Error("alice_esd", "cannot access esd " );
   } else {
       printf("*** accessed esd *** \n");  
   }


   // Set the branch addresses.
      
   printf("*** start setting branch addresses  ***\n");
   {

      TIter next(esd->fESDObjects);
      TObject* el ;
      while ((el=(TNamed*)next()))
      {
          TString bname(el->GetName());
          printf("*** bname %s \n" , bname.Data() );
          if(bname.CompareTo("AliESDfriend")==0) 
          {  
            // AliESDfriend needs some '.' magick.
            printf("*** ESDfriend. magic \n");
            esd_tree->SetBranchAddress("ESDfriend.", esd->fESDObjects->GetObjectRef(el));
          } else {
            printf("*** normal branch  \n");
            esd_tree->SetBranchAddress(bname, esd->fESDObjects->GetObjectRef(el));
          }
      }
   }

   printf("*** done setting branch addresses  ***\n");

}



//______________________________________________________________________________
Bool_t AliEvReader::LoadProject(const char* file, const char* project)
{
   // formerly alice_esd_loadlib
   // Make sure that shared library created from the auto-generated project
   // files exists and load it.

   TFile::SetCacheFileDir(".");
   TString lib(Form("%s/%s.%s", project, project, gSystem->GetSoExt()));
   printf("*** AliEvReader::LoadProject lib:%s file:%s  ***\n", lib.Data(), file );
   

   if (gSystem->AccessPathName(lib, kReadPermission)) {
      TFile* f = TFile::Open(file, "CACHEREAD");
      if (f == 0){
         return kFALSE;
      }
      f->MakeProject(project, "*", "++");
      f->Close();
      delete f;
   } else {
     printf("*** AliEvReader::LoadProject failed to access " ); 
   }
   return gSystem->Load(lib) >= 0;
}


Bool_t AliEvReader::LoadGeometry()
{
    LoadGeometry( esd_geom_file_name , esd_geom_name );
}


Bool_t AliEvReader::LoadGeometry(const char* file , const char* shape )
{

     // Simple geometry
      TFile::SetCacheFileDir(".");
      printf("AliEvReader::LoadGeometry %s %s \n" , file, shape );

      TFile* geom = TFile::Open( file , "CACHEREAD");
      if (!geom)
         return;
      TEveGeoShapeExtract* gse = (TEveGeoShapeExtract*) geom->Get(shape);
      gGeoShape = TEveGeoShape::ImportShapeExtract(gse, 0);
      geom->Close();
      delete geom;
      gEve->AddGlobalElement(gGeoShape);
}



//______________________________________________________________________________
void AliEvReader::Read()   
{
   // formerly alice_esd_read
   // Read tracks and associated clusters from current event.

   AliESDRun    *esdrun = (AliESDRun*)    esd->fESDObjects->FindObject("AliESDRun");
   TClonesArray *tracks = (TClonesArray*) esd->fESDObjects->FindObject("Tracks");

   // This needs further investigation. Clusters not shown.
   // AliESDfriend *frnd   = (AliESDfriend*) esd->fESDObjects->FindObject("AliESDfriend");
   // printf("Friend %p, n_tracks:%d\n", frnd, frnd->fTracks.GetEntries());


   EvNav* evn = EvNav::GetEvNav();
   TEveTrackList* track_list = evn->GetTrackList() ;

   if (track_list == 0) {
      printf("AliEvReader::Read creating new track_list " );
      track_list = new TEveTrackList("ESD Tracks"); 
      track_list->SetMainColor(6);
      //track_list->SetLineWidth(2);
      track_list->SetMarkerColor(kYellow);
      track_list->SetMarkerStyle(4);
      track_list->SetMarkerSize(0.5);

      gEve->AddElement(track_list);
   } else {
      printf("AliEvReader::Read reusing track_list " );
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
TEveTrack* AliEvReader::esd_make_track(TEveTrackPropagator*   trkProp,
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
Bool_t AliEvReader::trackIsOn(AliESDtrack* t, Int_t mask)
{
   // Check is track-flag specified by mask are set.

   return (t->fFlags & mask) > 0;
}

//______________________________________________________________________________
void AliEvReader::trackGetPos(AliExternalTrackParam* tp, Double_t r[3])
{
   // Get global position of starting point of tp.

  r[0] = tp->fX; r[1] = tp->fP[0]; r[2] = tp->fP[1];

  Double_t cs=TMath::Cos(tp->fAlpha), sn=TMath::Sin(tp->fAlpha), x=r[0];
  r[0] = x*cs - r[1]*sn; r[1] = x*sn + r[1]*cs;
}

//______________________________________________________________________________
void AliEvReader::trackGetMomentum(AliExternalTrackParam* tp, Double_t p[3])
{
   // Return global momentum vector of starting point of tp.

   p[0] = tp->fP[4]; p[1] = tp->fP[2]; p[2] = tp->fP[3];

   Double_t pt=1./TMath::Abs(p[0]);
   Double_t cs=TMath::Cos(tp->fAlpha), sn=TMath::Sin(tp->fAlpha);
   Double_t r=TMath::Sqrt(1 - p[1]*p[1]);
   p[0]=pt*(r*cs - p[1]*sn); p[1]=pt*(p[1]*cs + r*sn); p[2]=pt*p[2];
}

//______________________________________________________________________________
Double_t AliEvReader::trackGetP(AliExternalTrackParam* tp)
{
   // Return magnitude of momentum of tp.

   return TMath::Sqrt(1.+ tp->fP[3]*tp->fP[3])/TMath::Abs(tp->fP[4]);
}




