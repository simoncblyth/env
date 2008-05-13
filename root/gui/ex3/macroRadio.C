
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


class SubFrame;

class MyFrame {

   RQ_OBJECT("MyFrame")

   private:
      TGMainFrame        *fMain;

   // MenuBar Frame
      TGPopupMenu        *fMenuFile;
      TGMenuBar          *fMenuBar;
      TGHorizontal3DLine *fLineH1;

   // SubFrame
      TGHorizontalFrame  *fHF1;
      SubFrame           *fSubFrame;

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

   private:
      void CreateMenuBar();
      void DeleteMenuBar();
      void CreateSubFrame(TGCompositeFrame *parent);
      void DeleteSubFrame();

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
};

//debug: print function names
const Bool_t kCS  = 1; 

#if !defined (__CINT__) || defined (__MAKECINT__)
ClassImp(MyFrame);
ClassImp(SubFrame);
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
   fHF1 = new TGHorizontalFrame(fMain, 20, 20);
   fMain->AddFrame(fHF1, fLayout);

// Create subframe
   CreateSubFrame(fHF1);

// Main settings
   fMain->SetWindowName("RadioGui");
   fMain->MapSubwindows();
   fMain->Resize(300,250);
   fMain->MapWindow();
   fMain->Move(40, 40);
}//Constructor

//______________________________________________________________________________
MyFrame::~MyFrame()
{
   if(kCS) cout << "------MyFrame::~MyFrame------" << endl;

   DeleteSubFrame();
   DeleteMenuBar();

   delete fHF1; 
   delete fMain;

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

// Create menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

// Add menus to MenuBar
   fMenuBar = new TGMenuBar(fMain, 1, 1, kHorizontalFrame);
   fMenuBar->AddPopup("&File",        fMenuFile, fMenuBarItemLayout);

   fMain->AddFrame(fMenuBar, fMenuBarLayout);

// Line to separate menubar
   fLineH1 = new TGHorizontal3DLine(fMain);
   fLineLayout = new TGLayoutHints(kLHintsTop | kLHintsExpandX);
   fMain->AddFrame(fLineH1, fLineLayout);
   fLineH1->DrawBorder();
}//CreateMenuBar

//______________________________________________________________________________
void MyFrame::DeleteMenuBar()
{
   if(kCS) cout << "------MyFrame::DeleteMenuBar------" << endl;

   delete fLineH1; 
   delete fMenuBar;
   delete fMenuFile; 

   delete fMenuBarLayout;
   delete fMenuBarItemLayout;
   delete fMenuBarHelpLayout;
   delete fLineLayout;
}//DeleteMenuBar

//______________________________________________________________________________
void MyFrame::CreateSubFrame(TGCompositeFrame *parent)
{
   if(kCS) cout << "------MyFrame::CreateSubFrame------" << endl;

   fSubFrame = new SubFrame(parent, this, 10, 10);
}//CreateSubFrame

//______________________________________________________________________________
void MyFrame::DeleteSubFrame()
{
   if(kCS) cout << "------MyFrame::DeleteSubFrame------" << endl;

   delete fSubFrame;
}//DeleteSubFrame

//______________________________________________________________________________
void MyFrame::DoCloseWindow()
{
   if(kCS) cout << "------MyFrame::DoCloseWindow------" << endl;

   delete this;  //does not exit root
//   gApplication->Terminate(0);  //exit root, needed for standalone App
}//DoCloseWindow


//______________________________________________________________________________
//______________________________________________________________________________
SubFrame::SubFrame(TGCompositeFrame *parent, MyFrame *main, UInt_t w, UInt_t h)
{
   if(kCS) cout << "------SubFrame::SubFrame------" << endl;

   TGLayoutHints *hint;

   fMyFrame = main;
   fTrash = new TList();

   fGC1 = new TGCompositeFrame(parent, w, h);
   hint = new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY);
   parent->AddFrame(fGC1, hint);
   fTrash->Add(hint);

   fFG1 = new TGGroupFrame(fGC1, "Group frame", kVerticalFrame);
   hint = new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY);
//   hint = new TGLayoutHints(kLHintsTop | kLHintsLeft, 2,2,2,2);
   fGC1->AddFrame(fFG1, hint);
   fTrash->Add(hint);
   fFG1->Resize(300, 250);

   fTab1 = new TGTab(fFG1, 20, 20);
   hint = new TGLayoutHints(kLHintsTop | kLHintsLeft | kFixedSize, 0, 0, 10, 0);
   fFG1->AddFrame(fTab1, hint);
   fTrash->Add(hint);

   // Create tabs
   CreateTabFrame(fTab1);
}//Constructor

//______________________________________________________________________________
SubFrame::~SubFrame()
{
   if(kCS) cout << "------SubFrame::~SubFrame------" << endl;

   DeleteTabFrame();

   delete fTab1; 
   delete fFG1; delete fGC1; 

   fTrash->Delete();
   delete fTrash;

   fMyFrame = 0;
}//Destructor

//______________________________________________________________________________
void SubFrame::CreateTabFrame(TGTab *tab)
{
   if(kCS) cout << "------SubFrame::CreateTabFrame------" << endl;

   TGLayoutHints    *hint     = 0;
   TGCompositeFrame *tabframe = 0;

// Tab Selector
   tabframe = tab->AddTab("MyTab");
   fCFTab = new TGCompositeFrame(tabframe, 60, 20, kVerticalFrame);
   hint = new TGLayoutHints(kLHintsTop | kLHintsLeft, 2, 2, 5, 2);
   tabframe->AddFrame(fCFTab, hint);
   fTrash->Add(hint);

   CreateRadioFrame(fCFTab);
}//CreateTabFrame

//______________________________________________________________________________
void SubFrame::DeleteTabFrame()
{
   if(kCS) cout << "------SubFrame::DeleteTabFrame------" << endl;

   DeleteRadioFrame();

   delete fCFTab;
}//DeleteTabFrame

//______________________________________________________________________________
void SubFrame::CreateRadioFrame(TGCompositeFrame *parent)
{
   if(kCS) cout << "------SubFrame::CreateRadioFrame------" << endl;

   TGTableLayout      *layout = 0;
   TGTableLayoutHints *thint  = 0;
   TGLayoutHints      *hint   = 0;

   fFGRadio = new TGGroupFrame(parent, "RadioFrame", kHorizontalFrame);
   hint = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 7, 7, 10, 10);
   parent->AddFrame(fFGRadio, hint);
   fTrash->Add(hint);

   fCFRad = new TGCompositeFrame(fFGRadio,100,40);
   layout = new TGTableLayout(fCFRad, 2, 2);
   fCFRad->SetLayoutManager(layout);  //layout deleted by fCFRad!
   hint  = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   fFGRadio->AddFrame(fCFRad, hint);
   fTrash->Add(hint);

   fLab1 = new TGLabel(fCFRad, new TGString("Test 1:"));
   thint = new TGTableLayoutHints(0,1, 0,1, kLHintsNormal, 5,5,15,5);
   fCFRad->AddFrame(fLab1,thint);
   fTrash->Add(thint);

   fRadioID = 21;
   fRad1 = new TGRadioButton(fCFRad, new TGHotString("Radio 1"), 21);
cout << "Begin Connect" << endl;
   fRad1->Connect("Pressed()", "SubFrame", this, "DoClickRadio()");
cout << "End Connect" << endl;
   fRad1->SetState(kButtonDown);
   thint = new TGTableLayoutHints(1,2, 0,1, kLHintsNormal, 5,5,15,2);
   fCFRad->AddFrame(fRad1, thint);
   fTrash->Add(thint);

   fLab2 = new TGLabel(fCFRad, new TGString("Test 2:"));
   thint = new TGTableLayoutHints(0,1, 1,2, kLHintsNormal, 5,5,15,5);
   fCFRad->AddFrame(fLab2,thint);
   fTrash->Add(thint);

   fRad2 = new TGRadioButton(fCFRad, new TGHotString("Radio 2"), 22);
   fRad2->Connect("Pressed()", "SubFrame", this, "DoClickRadio()");
   fRad2->SetState(kButtonDown);
   thint = new TGTableLayoutHints(1,2, 1,2, kLHintsNormal, 5,5,15,2);
   fCFRad->AddFrame(fRad2, thint);
   fTrash->Add(thint);

   fFGRadio->MapSubwindows();
   fFGRadio->Layout();
}//CreateRadioFrame

//______________________________________________________________________________
void SubFrame::DeleteRadioFrame()
{
   if(kCS) cout << "------SubFrame::DeleteRadioFrame------" << endl;

   delete fRad2; delete fLab2;
   delete fRad1; delete fLab1;
   delete fCFRad; delete fFGRadio;
}//DeleteRadioFrame

//______________________________________________________________________________
void SubFrame::DoClickRadio(Int_t id)
{
   if(kCS) cout << "------SubFrame::DoClickRadio------" << endl;

   if (id == -1) {
      TGButton *btn = (TGButton*)gTQSender;
      id  = btn->WidgetId();
   }//if

   if (id == fRadioID) return;
   fRadioID = id;

   switch (id) {
      case 21:
         printf("Radio 1\n");
         fRad1->SetState(kButtonDown);
         fRad2->SetState(kButtonUp);
         break;

      case 22:
         printf("Radio 2\n");
         fRad1->SetState(kButtonUp);
         fRad2->SetState(kButtonDown);
         break;

      default:
         printf("Error: <SubFrame::DoClickRadio> Unknown ID %d selected\n", id);
         break;
   }//switch
}//DoClickRadio

//______________________________________________________________________________
void macroRadio()
{
   new MyFrame(gClient->GetRoot(), 400, 220);
}

