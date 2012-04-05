
#include <iostream>
#include <string>
#include <cstdlib>

#include "element.hh"
#include "model.hh"

#include "dbxml/DbXml.hpp"

using namespace std ;
using namespace DbXml ;


const char* Element::BlankName = "" ;

// local names of container elements "rez:quote" which contain the leaves
const char* Element::QuoteName = "quote" ;  
const char* Element::ModeName  = "mode" ;
const char* Element::ValueName = "value" ;  
const char* Element::ErrName   = "err" ;
const char* Element::XErrName  = "xerr" ;


// leaves of "rez:value"
const char* Element::VNumberName  = "number" ;
const char* Element::VUnitName    = "unit" ;
const char* Element::VLimitName   = "limit" ;
const char* Element::VCLName      = "cl" ;

// leaves of "rez:err" or "rez:xerr"
const char* Element::ENameName    = "errname" ;
const char* Element::ETypeName    = "type" ;
const char* Element::EPlusName    = "plus" ;
const char* Element::EMinusName   = "minus" ;

// leaves of "rez:mode"
const char* Element::MSrcName   = "src" ;
const char* Element::MProdName  = "prod" ;
const char* Element::MQwnName  =  "qwn" ;
const char* Element::MTagName  =  "mtag" ;

// leaves of rez:quote
const char* Element::QStatusName  = "status" ;  
const char* Element::QTitleName   = "title" ;  
const char* Element::QCommentName = "comment" ;  
const char* Element::QTagName     = "qtag" ;  



Element::Type Element::elementType( const unsigned char* localName, Element::Type regn )
{
   // element names are always ascii so presumably casting away "unsigned" is OK
   const char* s = (const char*)localName ; 

   if( regn == Element::Undefined ){

	// always check for container elem to allows regn switching    
        if(          strcmp(s, Element::QuoteName ) == 0){ return Element::Quote ; }
	else if(     strcmp(s, Element::ModeName )  == 0){ return Element::Mode  ; }
	else if(     strcmp(s, Element::ValueName ) == 0){ return Element::Value ; }
	else if(     strcmp(s, Element::ErrName )   == 0){ return Element::Err   ; }
	else if(     strcmp(s, Element::XErrName )  == 0){ return Element::XErr  ; }
        return  Element::Undefined ;

   } else if( regn == Element::Quote ){	   

	if(     strcmp(s, Element::QStatusName )   == 0){ return Element::QStatus  ; }
        else if(strcmp(s, Element::QTitleName )    == 0){ return Element::QTitle   ; }
        else if(strcmp(s, Element::QCommentName )  == 0){ return Element::QComment ; }
        else if(strcmp(s, Element::QTagName )      == 0){ return Element::QTag     ; }

        else if(strcmp(s, Element::ModeName )     == 0){ return Element::Mode  ; }
        else if(strcmp(s, Element::ValueName )    == 0){ return Element::Value ; }
        else if(strcmp(s, Element::ErrName )      == 0){ return Element::Err   ; }
        else if(strcmp(s, Element::XErrName )     == 0){ return Element::XErr  ; }
        return  Element::Undefined ;

   } else if( regn == Element::Mode ){	   

        if(     strcmp(s, Element::MSrcName )    == 0){ return Element::MSrc    ; }
        else if(strcmp(s, Element::MProdName )   == 0){ return Element::MProd   ; }
        else if(strcmp(s, Element::MQwnName )    == 0){ return Element::MQwn    ; }
        else if(strcmp(s, Element::MTagName )    == 0){ return Element::MTag    ; }
        return  Element::Undefined ;

   } else if( regn == Element::Value ){	   

        if(     strcmp(s, Element::VNumberName )  == 0){ return Element::VNumber  ; }
        else if(strcmp(s, Element::VUnitName )    == 0){ return Element::VUnit    ; }
        else if(strcmp(s, Element::VLimitName )   == 0){ return Element::VLimit   ; }
        else if(strcmp(s, Element::VCLName )      == 0){ return Element::VCL      ; }
        return Element::Undefined ; 

   } else if( regn == Element::Err || regn == Element::XErr  ){	   

        if(     strcmp(s, Element::ENameName )  == 0){ return Element::EName    ; }
        else if(strcmp(s, Element::ETypeName )  == 0){ return Element::EType    ; }
        else if(strcmp(s, Element::EPlusName )  == 0){ return Element::EPlus    ; }
        else if(strcmp(s, Element::EMinusName ) == 0){ return Element::EMinus   ; }
        return Element::Undefined ; 

   } 
   return Element::Undefined ; 
}


void Element::read(XmlEventReader& rdr, Quo& q )
{
   Element::Type regn(Element::Undefined);
   Element::Type curr(Element::Undefined);

   // traverse Xml taking note of leaf container elements "rez:mode" "rez:value" "rez:err" "rez:xerr"
   // and parsing text 

   while (rdr.hasNext()) {
	XmlEventReader::XmlEventType type = rdr.next();
	if (type == XmlEventReader::StartElement) 
	{
            if (!rdr.isEmptyElement()){
		curr = Element::elementType( rdr.getLocalName(), regn );
		if(
		    curr == Element::Quote || 
		    curr == Element::Mode  || 
		    curr == Element::Value || 
		    curr == Element::Err   || 
		    curr == Element::XErr    ) regn = curr ; 
            }		    
        } 
	else if ( type == XmlEventReader::Characters )
	{
	    size_t len ;	
            const unsigned char* value = rdr.getValue(len);
            const char* blnk = "" ;
	    string text = (rdr.isWhiteSpace()) ? blnk : reinterpret_cast<const char*>(value)  ; 	    
            parse(q, regn, curr, text );           
	}
	else if ( type == XmlEventReader::EndElement )
	{
	    curr = Element::Undefined ;	  // maybe should reset to prior regn
	}
   }	
}


void Element::fillDouble( const char* s , double& out )
{
    out  =  strtod(s, 0);       
}	
void Element::fillInt( const char* s , int& out )
{
    long int il =  strtol(s, NULL,  0);       
    out  =  (int)il ;
    // expect to never overflow int with PDG codes
}	
void Element::fillIntv( const char* s , vector<int>& out )
{
    long int il = strtol(s, NULL,  0);
    out.push_back((int)il);   
}	
void Element::fillString( const char* s , string& out )
{
    out  =  s ;       
}	


void Element::parse( Quo& q, Element::Type regn , Element::Type curr , string& text )
{
    if(regn == Element::Undefined ) return ;
    cout << "Element::parse " << regn << " : " << curr << " : " << text << endl;

    const char* s = text.c_str(); 
    if(regn == Element::Mode )
    {
	vector<Factor>& facv = q._factor ;    
        if(curr == regn) facv.push_back( Factor() ) ;     
	Factor& fac = facv.back() ;    
        switch(curr)
	{
	    case Element::MSrc:    fillInt(   s, fac._src)  ; break ;
	    case Element::MProd:   fillIntv(  s, fac._prod) ; break ;
	    case Element::MQwn:    fillString(s, fac._qwn)  ; break ;
	    case Element::MTag:    fillString(s, fac._tag)  ; break ;
	}	 
    }	    
    else if(regn == Element::Value )
    {	   
	Val& val = q._val ;    
        switch(curr)
	{
	    case Element::VNumber:  fillDouble(s, val._number) ; break ;
	    case Element::VUnit:    fillDouble(s, val._unit)   ; break ;
	    case Element::VLimit:   fillDouble(s, val._limit)  ; break ;
	}	 

    }
    else if( regn == Element::Err || regn == Element::XErr )
    {	    
        vector<Err>& errv = ( regn == Element::Err ) ? q._err : q._xerr ;  // refer to the appropriate vector of the 2
        if(curr == regn) errv.push_back( Err() ) ;                          // on entering "top-level" elements, mint a new Err 
	Err& err = errv.back();                                             // refer to last Err in the vector
        switch(curr)
	{
	    case Element::EPlus:   fillDouble(s, err._plus    ) ; break ;
	    case Element::EMinus:  fillDouble(s, err._minus   ) ; break ;
	    case Element::EName:   fillString(s, err._errname ) ; break ;
	}	
    }

}


