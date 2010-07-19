
///////////////////////////////////////////////////////////////////////
//
// {{ cls }}
//
// Package: Dbi (Database Interface).
//
//
// Concept: A concrete data type corresponding to a single row in a
//          database table of non-aggregated data.
//   
// Purpose: 
//
///////////////////////////////////////////////////////////////////////

#include "DbiDataSvc/{{ cls }}.h"
#include "DatabaseInterface/DbiLog.h"
#include "DatabaseInterface/DbiOutRowStream.h"
#include "DatabaseInterface/DbiResultSet.h"
#include "DatabaseInterface/DbiValidityRec.h"

//   Definition of static data members
//   *********************************


//  Instantiate associated Result Pointer and writer classes.
//  ********************************************************

#include "DatabaseInterface/DbiResultPtr.tpl"
template class  DbiResultPtr<{{ cls }}>;

#include "DatabaseInterface/DbiWriter.tpl"
template class  DbiWriter<{{ cls }}>;

// Definition of member functions (alphabetical order)
// ***************************************************

//.....................................................................

void {{ cls }}::Fill(DbiResultSet& rs,
                        const DbiValidityRec* /* vrec */) {
//
//
//  Purpose:  Fill object from Result Set
//
//  Arguments: 
//    rs           in    Result Set used to fill object
//    vrec         in    Associated validity record (or 0 if filling
//                                                    DbiValidityRec)
//  Return:    
//
//
//  Specification:-
//  =============
//
//  o Fill object from current row of Result Set.

//  Program Notes:-
//  =============

  Int_t numCol = rs.NumCols();
  //  The first column (SeqNo) has already been processed.
  for (Int_t curCol = 2; curCol <= numCol; ++curCol) {
    string colName = rs.CurColName();
    {% for r in t %}
    {% if forloop.first %}{% else %}else {% endif %}if ( colName == "{{ r.name }}" ){% if r.code2db %}
      {
         int {{ r.name }} = 0;  rs >> {{ r.name }} ;
         m_{{ r.name }} = {{ r.codetype }}( {{ r.name }} );
      }{% else %} rs >> m_{{ r.name }} ; {% endif %}{% endfor %}
      else 
      {
           LOG(dbi,Logging::kDebug1) << "Ignoring column " << curCol 
              << "(" << colName << ")"
              << "; not part of {{ cls }}" << std::endl;
              rs.IncrementCurCol();
      }
  }
}


//.....................................................................

void {{ cls }}::Store(DbiOutRowStream& ors,
                         const DbiValidityRec* /* vrec */) const {
//
//
//  Purpose:  Stream object to output row stream
//
//  Arguments: 
//    ors          in     Output row stream.
//    vrec         in    Associated validity record (or 0 if filling
//                                                    DbiValidityRec)
//
//  Return:    
//
//
//  Specification:-
//  =============
//
//  o  Stream object to output row stream.

//  Program Notes:-
//  =============

//  None.


  {% for r in t %}{% if forloop.first %}ors {% else %}    {% endif %} << m_{{r.name}}{%if r.code2db %}{{ r.code2db }}{% endif %}{% if forloop.last %};{% endif %}
  {% endfor %}

  //  m_describ was skipped in the hand-crafted SimPmtSpec.cc 
}

