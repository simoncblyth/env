
class EvNav
{
public:
   static EvNav* GetEvNav();
   static EvNav* gEvNav ;

   void Fwd();
   void Bck();
   void load_event();

   TEveTrackList* GetTrackList();
   Int_t GetEventId();

private:
    void EvNav() ;
    Int_t esd_event_id  ;  // esd_ is historical ... to be changed
    TEveTrackList* track_list ;
};


EvNav* EvNav::gEvNav = 0 ;


void EvNav::EvNav() 
{
      esd_event_id = 0 ;
      track_list = 0 ;
}


EvNav* EvNav::GetEvNav()
{
      if(gEvNav == 0 ) gEvNav = new EvNav ;
      return gEvNav ;
}

void EvNav::Fwd()
{
      if (esd_event_id < esd_tree->GetEntries() - 1) {
         ++esd_event_id;
         EvNav::load_event();

         evd = EvDisp::GetEvDisp();
         evd->update_projections();

      } else {
         gTextEntry->SetTextColor(0xff0000);
         gTextEntry->SetText("Already at last event");
         printf("Already at last event.\n");
      }
}

void EvNav::Bck()
{
      if (esd_event_id > 0) {
         --esd_event_id;
         EvNav::load_event();

         evd = EvDisp::GetEvDisp();
         evd->update_projections();


      } else {
         gTextEntry->SetTextColor(0xff0000);
         gTextEntry->SetText("Already at first event");
         printf("Already at first event.\n");
      }
}

void EvNav::load_event()
{
   // Load event specified in global esd_event_id.
   // The contents of previous event are removed.

      printf("Loading event %d.\n", esd_event_id);
      gTextEntry->SetTextColor(0xff0000);
      gTextEntry->SetText(Form("Loading event %d...",esd_event_id));
      gSystem->ProcessEvents();

      IEvReader* er = EvReader::GetEvReader();

      if (track_list){
           printf("load_event clearing track_list " );
           track_list->DestroyElements();
      } 

      esd_tree->GetEntry(esd_event_id);
      er->Read();

      gEve->Redraw3D(kFALSE, kTRUE);
      gTextEntry->SetTextColor(0x000000);
      gTextEntry->SetText(Form("Event %d loaded",esd_event_id));
      gROOT->ProcessLine("SplitGLView::UpdateSummary()");
}


TEveTrackList* EvNav::GetTrackList(){ 
       return track_list ; 
}

Int_t EvNav::GetEventId(){            
       return esd_event_id ;  
}



