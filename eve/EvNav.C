/*

    EvNav 
        uses EvDisp and EvReader


*/

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
         load_event();

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
         load_event();

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


      Int_t eid          = GetEventId();
      TEveTrackList* tkl = GetTrackList();
      
      printf("Loading event %d %x .\n", eid, tkl );
      gTextEntry->SetTextColor(0xff0000);
      gTextEntry->SetText(Form("Loading event %d %x...\n",eid, tkl));
      gSystem->ProcessEvents();

      IEvReader* er = EvReader::GetEvReader();


      if (tkl == 0){
           printf("EvNav::load_event NOT clearing tkl %x \n", tkl );
      } else {
           printf("EvNav::load_event clearing tkl %x \n", tkl );
           tkl->DestroyElements();
      }

      esd_tree->GetEntry(eid);
      er->Read();

      gEve->Redraw3D(kFALSE, kTRUE);
      gTextEntry->SetTextColor(0x000000);
      gTextEntry->SetText(Form("Event %d loaded",eid));
      gROOT->ProcessLine("SplitGLView::UpdateSummary()");
}


TEveTrackList* EvNav::GetTrackList(){ 
       return track_list ; 
}

void EvNav::SetTrackList( TEveTrackList* tkl ){ 
       track_list = tkl ; 
}



Int_t EvNav::GetEventId(){            
       return esd_event_id ;  
}






