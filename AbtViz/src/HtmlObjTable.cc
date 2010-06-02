// Lifted from $ROOSYS/tutorials/eve/SplitGLView.C 
// at ROOT version 5.21.04
#include "HtmlObjTable.h"

ClassImp(HtmlObjTable)

//______________________________________________________________________________
HtmlObjTable::HtmlObjTable(const char *name, Int_t nfields, Int_t nvals, Bool_t exp, Bool_t slab, Bool_t sval, Bool_t cbox, const char* fmt ) : 
   fName(name), fNValues(nvals), fNFields(nfields), fExpand(exp), fSideLab(slab), fSideVal(sval), fCheckBox(cbox), fFmt(fmt)
{
   // Constructor.

   fValues = new TArrayF[fNFields];
   for (int i=0;i<fNFields;i++)
      fValues[i].Set(nvals);
   fLabels = new TString[fNFields];
   fSideLabels = new TString[fNValues];
   fSideValues = new TString[fNValues];
}


//______________________________________________________________________________
HtmlObjTable::~HtmlObjTable()
{
   // Destructor.

   delete [] fValues;
   delete [] fLabels;
   delete [] fSideLabels;
   delete [] fSideValues;
}

//______________________________________________________________________________
void HtmlObjTable::Build()
{
   // Build HTML code.

   fHtml = "<table width=100% border=1 cellspacing=0 cellpadding=0 bgcolor=f0f0f0> ",

   BuildTitle();
   if (fExpand && (fNFields > 0 || fNValues > 0) ) {
      BuildLabels();
      BuildTable();
   }
   fHtml += "</table>";
}

//______________________________________________________________________________
void HtmlObjTable::BuildTitle()
{
   // Build table title.
   
   fHtml += "<tr><td colspan=";
  
   Int_t n = fNFields ;
   if( fCheckBox ) n++ ;
   if( fSideLab  ) n++ ;
   if( fSideVal  ) n++ ;
   fHtml += Form("%d>", n );

   fHtml += "<table width=100% border=0 cellspacing=2 cellpadding=0 bgcolor=6e6ea0>";
   fHtml += "<tr><td align=left>";
   fHtml += "<font face=Verdana size=3 color=ffffff><b><i>";
   fHtml += fName;
   fHtml += "</i></b></font></td>";
   fHtml += "<td>";
   fHtml += "<td align=right> ";
   fHtml += "<font face=Verdana size=3 color=ffffff><b><i>";
   fHtml += Form("Size = %d", fNValues);
   fHtml += "</i></b></font></td></tr>";
   fHtml += "</table>";
   fHtml += "</td></tr>";
}

//______________________________________________________________________________
void HtmlObjTable::BuildLabels()
{
   // Build table labels.

   Int_t i;
   fHtml += "<tr bgcolor=c0c0ff>";
   if(fCheckBox) fHtml += "<th> </th>"; 
   if(fSideLab)  fHtml += "<th> </th>";
   if(fSideVal)  fHtml += "<th> </th>";
   for (i=0;i<fNFields;i++) {
      fHtml += "<th> ";
      fHtml += fLabels[i];
      fHtml += " </th>"; // for the top labels 
   }
   fHtml += "</tr>";
}

//______________________________________________________________________________
void HtmlObjTable::BuildTable()
{
   // Build part of table with values.

   //Printf("BuildTable %d " , fNValues );

   for (int i = 0; i < fNValues; i++) {
      if (i%2)
         fHtml += "<tr bgcolor=e0e0ff>";
      else
         fHtml += "<tr bgcolor=ffffff>";
      
      TString name = fName;
      name.ReplaceAll(" ", "_");
      // checkboxes
      if(fCheckBox){
         fHtml += "<td bgcolor=d0d0ff align=\"center\">";
         fHtml += "<input type=\"checkbox\" name=\"";
         fHtml += name;
         fHtml += Form("[%d]\">",i);
         fHtml += "</td>";
      }

      if(fSideLab){
          fHtml += "<td width=";
          fHtml += Form("%d%%", 30 );
          fHtml += " align=\"center\"";
          fHtml += ">";
          fHtml += Form( "%s" , fSideLabels[i].Data() );
          fHtml += "</td>";
      } 
      if(fSideVal){
          fHtml += "<td width=";
          fHtml += Form("%d%%", 30 );
          fHtml += " align=\"center\"";
          fHtml += ">";
          fHtml += Form( "%s" , fSideValues[i].Data() );
          fHtml += "</td>";
      } 

      for (int j = 0; j < fNFields; j++) {
         fHtml += "<td width=";
         fHtml += Form("%d%%", 100/fNFields);
         fHtml += " align=\"center\"";
         fHtml += ">";
         fHtml += Form( fFmt.Data() , fValues[j][i]);
         fHtml += "</td>";
      }
      fHtml += "</tr> ";
   }
}


