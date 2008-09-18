//
//   use from interactive root with 
//      .x guitest0.C
//
//  example from
//    http://root.cern.ch/root/roottalk/roottalk99/1114.html
//
//    
//   it fails 
//




#include <iostream.h>
 
int ex1(){
 
   //TGMainFrame *rMFrm = new TGMainFrame(gClient->GetRoot(), 500, 500);
   MyMainFrame* mainWin = new MyMainFrame(gClient->GetRoot(), 200, 220);
 
   TGCompositeFrame *rCFrm =
     new TGCompositeFrame(rMFrm, 50, 50, kHorizontalFrame|kSunkenFrame );
   TGButton *rPBut =
     new TGPictureButton( rMFrm, gClient->GetPicture("beeravatar.xpm"),
     "rAction();", 1);
 
   rPBut->Associate(rMFrm);
 
   rCFrm->AddFrame(rPBut, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   rMFrm->AddFrame(rCFrm, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   rMFrm->SetWindowName("First Rado's ROOT GUI");
   rMFrm->MapSubwindows();
   rMFrm->Resize(rMFrm->GetDefaultSize());
   rMFrm->MapWindow();
 
   return 0;
}

