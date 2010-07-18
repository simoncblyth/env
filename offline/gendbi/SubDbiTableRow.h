#ifndef {{ cls|upper }}_H
#define {{ cls|upper }}_H

////////////////////////////////////////////////////////////////////////
// {{ cls }}                                                          //
//                                                                    //
// Package: Dbi (Database Interface).                                 //
//                                                                    //
// Concept:  A concrete data type corresponding to a single row in    //
//           the {{ cls }}  database table of non-aggregated data.  //
//                                                                    //
////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"
#include "DatabaseInterface/DbiTableRow.h"
#include "DatabaseInterface/DbiLog.h"
#include "DatabaseInterface/DbiOutRowStream.h"
#include "DatabaseInterface/DbiResultSet.h"
#include "DatabaseInterface/DbiValidityRec.h"
#include "DatabaseInterface/DbiResultPtr.h"
#include "DataSvc/ICalibDataSvc.h"
#include "Conventions/Detectors.h"
#include <string>

using namespace std;

class DbiValidityRec;

class {{ cls }} : public DbiTableRow
{
public:
  {{ cls }}(){}
  {{ cls }}(const {{ cls }}& from) : DbiTableRow(from) {  *this = from; }

  {{ cls }}(
    {% for r in t %}{{ r.codetype }} {{ r.name }}{% if forloop.last %} {% else %},{% endif %} // {{ r.description }}
    {% endfor %}
	      ) 
	  {
		 {% for r in t %} m_{{ r.name }} = {{ r.name }};
         {% endfor %} 
      }

  virtual ~{{ cls }}(){};

// State testing member functions
  Bool_t CanL2Cache() const { return {{ t.meta.CanL2Cache }}; }
  Bool_t Compare(const {{ cls }}& that ) const {
   {% for r in t %}{% if forloop.first %} return {% else %} && {% endif %}m_{{ r.name }} == that.m_{{ r.name }}
   {% endfor %}
;}

// Getters 
  {% for r in t %}{{ r.codetype }} Get{{ r.name|capfirst }}() const {return m_{{ r.name }}; } 
  {% endfor %}

  virtual DbiTableRow* CreateTableRow() const { return new {{ cls }} ; }
// I/O  member functions
  virtual void Fill(DbiResultSet& rs, const DbiValidityRec* vrec);
  virtual void Store(DbiOutRowStream& ors, const DbiValidityRec* vrec) const;

private:  

// Data members
  {% for r in t %}{{ r.codetype }} m_{{ r.name }}; // {{ r.description }}  
  {% endfor %}

};


#endif  // {{cls|upper}}_H


