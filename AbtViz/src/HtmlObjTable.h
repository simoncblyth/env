// Lifted from $ROOSYS/tutorials/eve/SplitGLView.C 
// at ROOT version 5.21.04

#ifndef htmlobjtable_h
#define htmlobjtable_h

#include "TObject.h"
#include "TString.h"
#include "TArrayF.h"

class HtmlObjTable : public TObject {
public:                     // make them public for shorter code

   TString   fName;
   Int_t     fNValues;      // number of values
   Int_t     fNFields;      // number of fields
   TArrayF  *fValues;
   TString  *fLabels;
   TString  *fSideLabels;
   TString  *fSideValues;
   Bool_t    fExpand;
   Bool_t    fSideLab;
   Bool_t    fSideVal;
   Bool_t    fCheckBox;
   TString   fFmt;

   TString   fHtml;         // HTML output code

   void Build();
   void BuildTitle();
   void BuildLabels();
   void BuildTable();

public:
   HtmlObjTable(const char *name, Int_t nfields, Int_t nvals, Bool_t exp=kTRUE, Bool_t slab=kFALSE, Bool_t sval=kFALSE, Bool_t cbox=kTRUE, const char* fmt="%1.1f" );
   virtual ~HtmlObjTable();
   void     SetLabel(Int_t col, const char *label) { fLabels[col] = label; }
   void     SetSideLabel(Int_t col, const char *label) { fSideLabels[col] = label; }
   void     SetSideValue(Int_t col, const char *value) { fSideValues[col] = value; }
   void     SetValue(Int_t col, Int_t row, Float_t val) { fValues[col].SetAt(val, row); }
   TString  Html() const { return fHtml; }

   ClassDef(HtmlObjTable, 0);
};

#endif

