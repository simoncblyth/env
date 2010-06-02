#ifndef htmlsummary_h
#define htmlsummary_h

#include "TString.h"
class TOrdCollection ;
class HtmlObjTable ;

class HtmlSummary {
public:                           // make them public for shorter code
   Int_t           fNTables;
   TOrdCollection *fObjTables;    // ->array of object tables
   TString         fHtml;         // output HTML string
   TString         fTitle;        // page title
   TString         fHeader;       // HTML header
   TString         fFooter;       // HTML footer

   void     MakeHeader(Bool_t titl=kTRUE);
   void     MakeFooter();

public:
   HtmlSummary(const char *title);
   virtual ~HtmlSummary();

   HtmlObjTable  *AddTable(const char *name, Int_t nfields, Int_t nvals, 
                           Bool_t exp=kTRUE, Bool_t slab=kFALSE, Bool_t sval=kFALSE, Bool_t cbox=kTRUE, Option_t *opt="");
   HtmlObjTable  *GetTable(Int_t at) const ;
   void           Build();
   void           Clear(Option_t *option="");
   void           Reset(Option_t *option="");
   TString        Html() const { return fHtml; }

   ClassDef(HtmlSummary, 0);
};


#endif

