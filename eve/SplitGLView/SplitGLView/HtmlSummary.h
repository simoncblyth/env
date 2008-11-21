
#include "TOrdCollection.h"
#include "TString.h"

class HtmlObjTable ;

class HtmlSummary {
public:                           // make them public for shorter code
   Int_t           fNTables;
   TOrdCollection *fObjTables;    // ->array of object tables
   TString         fHtml;         // output HTML string
   TString         fTitle;        // page title
   TString         fHeader;       // HTML header
   TString         fFooter;       // HTML footer

   void     MakeHeader();
   void     MakeFooter();

public:
   HtmlSummary(const char *title);
   virtual ~HtmlSummary();

   HtmlObjTable  *AddTable(const char *name, Int_t nfields, Int_t nvals, 
                           Bool_t exp=kTRUE, Option_t *opt="");
   HtmlObjTable  *GetTable(Int_t at) const { return (HtmlObjTable *)fObjTables->At(at); }
   void           Build();
   void           Clear(Option_t *option="");
   void           Reset(Option_t *option="");
   TString        Html() const { return fHtml; }

//   ClassDef(HtmlSummary, 0);
};



