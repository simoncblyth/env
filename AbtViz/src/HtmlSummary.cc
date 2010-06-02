
#include "HtmlSummary.h"
#include "HtmlObjTable.h"
#include "TOrdCollection.h"


ClassImp(HtmlSummary)

HtmlSummary::HtmlSummary(const char *title) : fNTables(0), fTitle(title)
{
   // Constructor.

   fObjTables = new TOrdCollection();
}
   
HtmlObjTable* HtmlSummary::GetTable(Int_t at) const { return (HtmlObjTable *)fObjTables->At(at); }

//______________________________________________________________________________
HtmlSummary::~HtmlSummary()
{
   // Destructor.

   Reset();
}

//______________________________________________________________________________
HtmlObjTable *HtmlSummary::AddTable(const char *name, Int_t nfields, Int_t nvals,
                                    Bool_t exp, Bool_t slab, Bool_t sval, Bool_t cbox, Option_t *option)
{
   // Add a new table in our list of tables.

   TString opt = option;
   opt.ToLower();
   HtmlObjTable *table = new HtmlObjTable(name, nfields, nvals, exp, slab, sval, cbox );
   fNTables++;
   if (opt.Contains("first"))
      fObjTables->AddFirst(table);
   else
      fObjTables->Add(table);
   return table;
}

//______________________________________________________________________________
void HtmlSummary::Clear(Option_t *option)
{
   // Clear the table list.

   if (option && option[0] == 'D')
      fObjTables->Delete(option);
   else
      fObjTables->Clear(option);
   fNTables = 0;
}

//______________________________________________________________________________
void HtmlSummary::Reset(Option_t *)
{
   // Reset (delete) the table list;

   delete fObjTables; fObjTables = 0;
   fNTables = 0;
}

//______________________________________________________________________________
void HtmlSummary::Build()
{
   // Build the summary.

   MakeHeader(kFALSE);
   for (int i=0;i<fNTables;i++) {
      GetTable(i)->Build();
      fHtml += GetTable(i)->Html();
   }
   //MakeFooter();
   fHtml += "</body></html>";
}

//______________________________________________________________________________
void HtmlSummary::MakeHeader(Bool_t titl)
{
   // Make HTML header.

   fHeader  = "<html><head><title>";
   fHeader += fTitle;
   fHeader += "</title></head><body>";
   if(titl){
       fHeader += "<center><h2><font color=#2222ee><i>";
       fHeader += fTitle;
       fHeader += "</i></font></h2></center>";
   }
   fHtml    = fHeader;
}

//______________________________________________________________________________
void HtmlSummary::MakeFooter()
{
   // Make HTML footer.

   fFooter  = "<br><p><br><center><strong><font size=2 color=#2222ee>";
   fFooter += "Example of using Html widget to display tabular data";
   fFooter += "<br>";
   fFooter += "© 2007-2008 Bertrand Bellenot";
   //fFooter += "</font></strong></center></body></html>";  
   fFooter += "</font></strong></center>";  
   fHtml   += fFooter;
}


