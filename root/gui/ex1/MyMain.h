
// File: MyMain.h

#include <TGClient.h>
#include <TGButton.h>
class MyMainFrame : public TGMainFrame {
private:
    TGTextButton    *fButton1, *fButton2;
    TGPictureButton *fPicBut;
    TGCheckButton   *fChkBut;
    TGRadioButton   *fRBut1, *fRBut2;
    TGLayoutHints   *fLayout;
public:
    MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
    ~MyMainFrame() { } // need to delete here created widgets
    Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
};

