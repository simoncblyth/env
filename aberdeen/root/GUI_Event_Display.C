
//
//  from http://darbujan.fzu.cz/~cint/03/msg00634.html
//  run from interactive root with
//     .x macroRadio.C  
// 



// File created: 09/11/2003                          last modified: 09/11/2003

/******************************************************************************
* Author: Christian Stratowa.                                                 *
******************************************************************************/

//#ifndef __CINT__
#include "RQ_OBJECT.h"
#include "TG3DLine.h"
#include "TGButton.h"
#include "TGFrame.h"
#include "TGMenu.h"
#include "TGLabel.h"
#include "TGLayout.h"
#include "TGTab.h"
#include "TGTableLayout.h"
#include "TList.h"
#include "TString.h"
//#endif
#include <Riostream.h>
#include "GeoMap.C"

Bool_t importb = 0;

class MyFrame {

   RQ_OBJECT("MyFrame")

//   private:
	public:
      TGMainFrame        *fMain;
      TRootEmbeddedCanvas *fEcanvas;

   // MenuBar Frame
      TGPopupMenu        *fMenuFile;
      TGPopupMenu	 *fMenuDisplay;
      TGPopupMenu	 *fMenuMac;
      TGMenuBar          *fMenuBar;
      TGHorizontal3DLine *fLineH1;

   // Layout hints
      TGLayoutHints      *fMenuBarLayout;
      TGLayoutHints      *fMenuBarItemLayout;
      TGLayoutHints      *fMenuBarHelpLayout;
      TGLayoutHints      *fLineLayout;
      TGLayoutHints      *fLayout;

      TList    *fTrash;

   public:
      MyFrame(const TGWindow *window, UInt_t w, UInt_t h);
      virtual ~MyFrame();

   // Slots
      void DoCloseWindow();
      void HandleMenu(Int_t id);

   // Macro
      void Import();
      void DoDraw();
      void Reset();

   // Func
      void CanvasUpdate(TRootEmbeddedCanvas* fEc);

   private:
      void CreateMenuBar();
      void DeleteMenuBar();


#if !defined (__CINT__) || defined (__MAKECINT__)
      ClassDef(MyFrame,0) //MainFrame
#endif
};

class SubFrame {

   RQ_OBJECT("SubFrame")

   private:
      MyFrame            *fMyFrame;
      TGCompositeFrame    *fGC1;
      TGGroupFrame         *fFG1;
      TGTab                 *fTab1;
      TGCompositeFrame       *fCFTab;
      TGGroupFrame            *fFGRadio;
      TGCompositeFrame         *fCFRad;
      TGLabel                   *fLab1;
      TGRadioButton             *fRad1;
      TGLabel                   *fLab2;
      TGRadioButton             *fRad2;

      TList    *fTrash;
      Int_t     fRadioID;

   public:
      SubFrame() {}
      SubFrame(TGCompositeFrame *parent, MyFrame *main, UInt_t w, UInt_t h);
      virtual ~SubFrame();

      void DoClickRadio(Int_t id = -1);

   private:
      void CreateTabFrame(TGTab *tab);
      void DeleteTabFrame();
      void CreateRadioFrame(TGCompositeFrame *parent);
      void DeleteRadioFrame();

#if !defined (__CINT__) || defined (__MAKECINT__)
      ClassDef(SubFrame,0) //SubFrame
#endif
};

// Menu commands
enum EMenuCommands {

   M_FILE,
   M_FILE_NEW,
   M_FILE_QUIT,

   M_IMPORT,
   M_DISPLAY,

   M_MAC_1,
   M_MAC_2,
};

//debug: print function names
const Bool_t kCS  = 1; 

#if !defined (__CINT__) || defined (__MAKECINT__)
ClassImp(MyFrame);
#endif

//______________________________________________________________________________
MyFrame::MyFrame(const TGWindow *window, UInt_t w, UInt_t h)
{
   if(kCS) cout << "------MyFrame::MyFrame------" << endl;

   fTrash = new TList();

   fMain = new TGMainFrame(window, w, h);
   fMain->Connect("CloseWindow()", "MyFrame", this, "DoCloseWindow()");

// Create menus
   CreateMenuBar();

// Basic frame layout
   fLayout = new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY);

// Create canvas widget, for containing Event Displat result
   fEcanvas = new TRootEmbeddedCanvas("Ecanvas",fMain,200,200);
   fMain->AddFrame(fEcanvas, fLayout);

// Main settings
   fMain->SetWindowName("Aberdeen Event Display");
   fMain->MapSubwindows();
   fMain->Resize(fMain->GetDefaultSize());
   fMain->MapWindow();
}//Constructor

//______________________________________________________________________________
MyFrame::~MyFrame()
{
   if(kCS) cout << "------MyFrame::~MyFrame------" << endl;

   DeleteMenuBar();

 
   delete fMain;
   delete fEcanvas;

   delete fLayout;

   fTrash->Delete();
   delete fTrash;
}//Destructor

//______________________________________________________________________________
void MyFrame::CreateMenuBar()
{
   if(kCS) cout << "------MyFrame::CreateMenuBar------" << endl;

// File menu
   fMenuFile = new TGPopupMenu(gClient->GetRoot());
   fMenuFile->AddEntry("&New...",  M_FILE_NEW);
   fMenuFile->AddEntry("&Quit...",  M_FILE_QUIT);

   fMenuDisplay = new TGPopupMenu(gClient->GetRoot());
   fMenuDisplay->AddEntry("&Import...", M_IMPORT);
   fMenuDisplay->AddEntry("&Dsplay", M_DISPLAY);

   fMenuMac = new TGPopupMenu(gClient->GetRoot());
   fMenuMac->AddEntry("&Test...", M_MAC_1);
   fMenuMac->AddEntry("&SecTest...", M_MAC_2);

// Create menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

// Add menus to MenuBar
   fMenuBar = new TGMenuBar(fMain, 1, 1, kHorizontalFrame);
   fMenuBar->AddPopup("&File",        fMenuFile, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Geometry",	fMenuDisplay, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Macro",fMenuMac, fMenuBarItemLayout);

   fMain->AddFrame(fMenuBar, fMenuBarLayout);

// Line to separate menubar
   fLineH1 = new TGHorizontal3DLine(fMain);
   fLineLayout = new TGLayoutHints(kLHintsTop | kLHintsExpandX);
   fMain->AddFrame(fLineH1, fLineLayout);
   fLineH1->DrawBorder();

// Handle the messages
   fMenuFile->Connect("Activated(Int_t)", "MyFrame", this,"HandleMenu(Int_t)");
   fMenuDisplay->Connect("Activated(Int_t)", "MyFrame", this,"HandleMenu(Int_t)");
   fMenuMac->Connect("Activated(Int_t)", "MyFrame", this,"HandleMenu(Int_t)");

}//CreateMenuBar

//------------------------------------------------------------------------------
void MyFrame::HandleMenu(Int_t id)
{
   // Handle the menu items
   switch (id) {


         case M_FILE_QUIT:
            CloseWindow();   // terminate theApp no need to use SendCloseMessage()
            break;

	 case M_IMPORT:
	    //gROOT->Reset();
//	    gROOT->ProcessLine(".L GeoMap.C");
//	    gROOT->ProcessLine(".x GUI_Import.C");
	    Import();
	    break;

	 case M_DISPLAY:
//	    gROOT->ProcessLine(".x GUI_Display.C");
	    DoDraw();
	    break;

	 case M_MAC_1:
	    cout << " Testing macro #1" << endl;
	    break;

         default:
            printf("Menu item %d selected\n", id);
            break;
   }
}

//______________________________________________________________________________
void MyFrame::DeleteMenuBar()
{
   if(kCS) cout << "------MyFrame::DeleteMenuBar------" << endl;

   delete fLineH1; 
   delete fMenuBar;
   delete fMenuFile;
   delete fMenuDisplay;
   delete fMenuMac;

   delete fMenuBarLayout;
   delete fMenuBarItemLayout;
   delete fMenuBarHelpLayout;
   delete fLineLayout;
}//DeleteMenuBar

//------------------------------------------------------------------------------
void MyFrame::CloseWindow()
{
   // Got close message for this MainFrame. Terminates the application.
   gApplication->Terminate();
}

//______________________________________________________________________________
void MyFrame::DoCloseWindow()
{
   if(kCS) cout << "------MyFrame::DoCloseWindow------" << endl;

   delete this;  //does not exit root
//   gApplication->Terminate(0);  //exit root, needed for standalone App
}//DoCloseWindow

//------------------------------------------------------------------------------
void MyFrame::Import() {
//  gROOT->Reset();
//  gROOT->ProcessLine(".L GeoMap.C");
  if(importb){
	gGeoManager->CloseGeometry();
	cout << " Closing exsiting Geometry and import new one" << endl;
//	gm(gClient->GetRoot(), 400, 220);
	gROOT->ProcessLine(".x GUI_Import.C");
  } else
  gROOT->ProcessLine(".x GUI_Import.C");
  CanvasUpdate(this->fEcanvas);
  
//  TCanvas* canvas = this->fEcanvas->GetCanvas();
//  TRootEmbeddedCanvas* c=this->fEcanvas;
//  TCanvas* canvas = c->GetCanvas();
//  canvas->Update();

  importb = 1;
}

void MyFrame::DoDraw() {
  gROOT->ProcessLine(".x GUI_Display.C");
  CanvasUpdate(this->fEcanvas);
}

void MyFrame::Reset() {
  gROOT->ProcessLine(".x GUI_Import.C");
  CanvasUpdate(this->fEcanvas);
}

//______________________________________________________________________________

void MyFrame::CanvasUpdate(TRootEmbeddedCanvas* fEc)
{
  TCanvas* c = fEc->GetCanvas();
  c->Update();

}

//______________________________________________________________________________
/*void GUI_Event_Display()
{
   new MyFrame(gClient->GetRoot(), 400, 220);
}
*/