#ifndef ROOT_TGShapedToolTip
#define ROOT_TGShapedToolTip

#include "TGShapedFrame.h"
#include "TString.h"

class TRootEmbeddedCanvas;
class TH1;
class TGLayoutHints;


class TGShapedToolTip : public TGShapedFrame {

private:
   TGShapedToolTip(const TGShapedToolTip&); // Not implemented
   TGShapedToolTip& operator=(const TGShapedToolTip&); // Not implemented

protected:
   Int_t                 fTextX, fTextY, fTextH;
   TString               fTextCol;

   TRootEmbeddedCanvas  *fEc;       // embedded canvas for histogram
   TH1                  *fHist;     // user histogram
   TString               fText;     // info (as tool tip) text

   virtual void          DoRedraw() {}

public:
   TGShapedToolTip(const char *picname, Int_t cx=0, Int_t cy=0, Int_t cw=0, 
                   Int_t ch=0, Int_t tx=0, Int_t ty=0, Int_t th=0, 
                   const char *col="#ffffff");
   virtual ~TGShapedToolTip();

   virtual void   CloseWindow();
   void           CreateCanvas(Int_t cx, Int_t cy, Int_t cw, Int_t ch);
   void           CreateCanvas(Int_t cw, Int_t ch, TGLayoutHints *hints);
   TH1           *GetHisto() const { return fHist; }
   const char    *GetText() const { return fText.Data(); }
   void           Refresh();
   void           SetHisto(TH1 *hist);
   void           SetText(const char *text);
   void           SetTextColor(const char *col);
   void           SetTextAttributes(Int_t tx, Int_t ty, Int_t th, const char *col=0);
   void           Show(Int_t x, Int_t y, const char *text = 0, TH1 *hist = 0);

   ClassDef(TGShapedToolTip, 0) // Shaped composite frame
};

#endif
