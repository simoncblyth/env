#ifndef ROOT_SplitGLView
#define ROOT_SplitGLView

#include "TGFrame.h"

class TEvePad;
class TGSplitFrame;
class TGLEmbeddedViewer;
class HtmlSummary;
class TGHtml;
class TGMenuBar;
class TGPopupMenu;
class TGStatusBar;
class TGShapedToolTip;
class TEveViewer;
class TEveProjectionManager;
class TGListTreeItem;
class TObject;
class TGLPhysicalShape;
class TGLViewerBase;

class SplitGLView : public TGMainFrame {

public:
   enum EMyCommands {
      kFileOpen, kFileExit, kFileLoadConfig, kFileSaveConfig,
      kHelpAbout, kGLPerspYOZ, kGLPerspXOZ, kGLPerspXOY, kGLXOY,
      kGLXOZ, kGLZOY, kGLOrthoRotate, kGLOrthoDolly, kSceneUpdate, 
      kSceneUpdateAll, kSummaryUpdate
   };

private:
   TEvePad               *fPad;           // pad used as geometry container
   TGSplitFrame          *fSplitFrame;    // main (first) split frame
   TGLEmbeddedViewer     *fViewer0;       // main GL viewer
   TGLEmbeddedViewer     *fViewer1;       // first GL viewer
   TGLEmbeddedViewer     *fViewer2;       // second GL viewer
   TGLEmbeddedViewer     *fActViewer;     // actual (active) GL viewer
   static HtmlSummary    *fgHtmlSummary;  // summary HTML table
   static TGHtml         *fgHtml;
   TGMenuBar             *fMenuBar;       // main menu bar
   TGPopupMenu           *fMenuFile;      // 'File' popup menu
   TGPopupMenu           *fMenuHelp;      // 'Help' popup menu
   TGPopupMenu           *fMenuCamera;    // 'Camera' popup menu
   TGPopupMenu           *fMenuScene;     // 'Scene' popup menu
   TGStatusBar           *fStatusBar;     // status bar
   TGShapedToolTip       *fShapedToolTip; // shaped tooltip
   Bool_t                 fIsEmbedded;

   TEveViewer            *fViewer[3];
   TEveProjectionManager *fRPhiMgr;
   TEveProjectionManager *fRhoZMgr;

public:
   SplitGLView(const TGWindow *p=0, UInt_t w=800, UInt_t h=600, Bool_t embed=kFALSE);
   virtual ~SplitGLView();

   void           ItemClicked(TGListTreeItem *item, Int_t btn, Int_t x, Int_t y);
   void           HandleMenu(Int_t id);
   void           OnClicked(TObject *obj);
   void           OnMouseIdle(TGLPhysicalShape *shape, UInt_t posx, UInt_t posy);
   void           OnMouseOver(TGLPhysicalShape *shape);
   void           OnViewerActivated();
   void           OpenFile(const char *fname);
   void           SwapToMainView(TGLViewerBase *viewer);
   void           ToggleOrthoRotate();
   void           ToggleOrthoDolly();
   void           UnDock(TGLViewerBase *viewer);
   void           LoadConfig(const char *fname);
   void           SaveConfig(const char *fname);
   static void    UpdateSummary();

   TEveProjectionManager *GetRPhiMgr() const { return fRPhiMgr; }
   TEveProjectionManager *GetRhoZMgr() const { return fRhoZMgr; }

   ClassDef(SplitGLView, 0)
};

// copying TSystem.h 
R__EXTERN TEveProjectionManager *gRPhiMgr ;
R__EXTERN TEveProjectionManager *gRhoZMgr ;

#endif
