
{

    // needs some work... the expanded menu hangs around all the time 
    
    
     // Bring up Eve GUI if not there already   ... creates gEve which ISA TEveManager* 
    if(TEveManager::gEve == NULL ){ 
        TEveManager::Create();       
    }
        
    TGWindow* fMainWindow = gEve->GetMainWindow() ;  // TGWindow* fClientRoot = gClient->GetRoot() ;  // not the same as fMainWindow  
    TGWindow* fMainFrameWindow = fMainWindow->GetMainFrame(); 
    TGMainFrame* fMain = (TGMainFrame*)fMainFrameWindow ;

    
    fMenuBar = new TGMenuBar( fMainWindow , 1, 1, kHorizontalFrame);
    fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
    fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);

    fMenu = new TGPopupMenu(fMainWindow) ;
    fMenu->AddEntry("&Next", 0);
    fMenu->AddEntry("&Prev", 1);
    fMenu->AddEntry("&Jump", 1);
    fMenu->AddEntry("&Load", 1);
      
    fMenuBar->AddPopup("&Event" , fMenu , fMenuBarItemLayout);

    fMain->AddFrame(fMenuBar, fMenuBarLayout); 
    fMain->MapSubwindows();
    fMain->Resize(fMain->GetDefaultSize());
    fMain->MapWindow();
 
}





