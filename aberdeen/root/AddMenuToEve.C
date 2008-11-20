
{

    //
    //   Adding Navigate menu at the top level directly into the GL Viewers 
    //   File/Camera/Help menu 
    //      ... nice user interface
    //      ... potential for keyboard control
    //      ... BUT  how to intercept the messages ? 
    //
    //     BUT associates it with the GL viewer only, could have other things like histos
    //         in added canvases, that could benefit from navigation controls ?
    //
    //      
    //
    //
    // root-
    // cd $ROOTSYS/tutorials/eve
    // root alice_esd.C 
    //  .x ~/env/aberdeen/root/AddMenuToEve.C
    //
    //
    //  TGMenu.cxx ...
    // Selecting a menu item will generate the event:                       
    // kC_COMMAND, kCM_MENU, menu id, user data.                            
    //
    //
    //
    // Bring up Eve GUI if not there already   ... creates gEve which ISA TEveManager* 
    if(TEveManager::gEve == NULL ){ 
        TEveManager::Create();       
    }
        
    TGMainFrame* fMain = (TGMainFrame*)gEve->GetMainWindow() ;  // TGWindow* fClientRoot = gClient->GetRoot() ;  // not the same as fMainWindow  

    // locate the GL Viewer File/Camera/Help menuBar and add a Navigate popup to it  
    TGLSAFrame* fFrame = ((TGLSAViewer*)gEve->GetGLViewer())->GetFrame(); 
    TGFrameElement* fe =  (TGFrameElement*)fFrame->GetList()->At(0);
    TGFrame* m = fe.fFrame ;

    if( m != NULL && !strcmp(m->ClassName(), "TGMenuBar")){

        fNavMenu = new TGPopupMenu(fFrame->GetClient()->GetRoot());
        fNavMenu->AddEntry("&Next", 0);
        fNavMenu->AddEntry("&Prev", 1);
        fNavMenu->AddEntry("&Jump", 2);
        fNavMenu->AddEntry("&Load", 3);

       // fNavMenu->Associate( w ); // sets the fMsgWindow, by default the parent 

        menuBar = (TGMenuBar*)m ;
        menuBar->AddPopup("&Navigate" , fNavMenu , new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0) );

    } else {
        cout << "Failed to find the GL Viewer File/Camera/Help menuBar " << endl ;
    }

    // redraw 
    fMain->MapSubwindows();
    fMain->Resize(fMain->GetDefaultSize());
    fMain->MapWindow();
 
}





