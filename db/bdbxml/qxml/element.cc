
#include <iostream>
#include <string>
#include <cstdlib>

#include "element.hh"
#include "model.hh"

#include "dbxml/DbXml.hpp"

using namespace std ;
using namespace DbXml ;


const char* Element::BlankName = "blank" ;

// local names of container elements "rez:quote" which contain the leaves
const char* Element::CHeaderName = "header" ;  
const char* Element::COriginName = "origin" ;  
const char* Element::CQuoteName  = "quote" ;  
const char* Element::CModeName   = "mode" ;
const char* Element::CValueName  = "value" ;  
const char* Element::CErrName    = "err" ;
const char* Element::CXErrName   = "xerr" ;


// leaves of "rez:value"
const char* Element::VNumberName  = "number" ;
const char* Element::VUnitName    = "unit" ;
const char* Element::VLimitName   = "limit" ;
const char* Element::VCLName      = "cl" ;

// leaves of "rez:err" or "rez:xerr"
const char* Element::ENameName    = "errname" ;
const char* Element::ETypeName    = "type" ;
const char* Element::ESymmName    = "symm" ;
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

// leaves of rez:header
const char* Element::HStatusName    = "status" ;  
const char* Element::HCategoryName  = "category" ;  
const char* Element::HNameName      = "name" ;  
const char* Element::HTitleName     = "title" ;  

// leaves of rez:origin
const char* Element::OUriName        = "uri" ;  
const char* Element::OGroupName      = "group" ;  
const char* Element::OOwnerName      = "owner" ;  
const char* Element::ODateTimeName   = "datetime" ;  


const char* Element::elementName( Element::Type type )
{
     switch(type)
     {
	     case CHeader:  return CHeaderName   ; break ;
	     case COrigin:  return COriginName   ; break ;
	     case CQuote:   return CQuoteName    ; break ;
	     case CMode:    return CModeName     ; break ;
	     case CValue:   return CValueName    ; break ;
	     case CErr:     return CErrName      ; break ;
	     case CXErr:    return CXErrName     ; break ;

	     case VNumber:  return VNumberName   ; break ;
	     case VUnit:    return VUnitName     ; break ;
	     case VLimit:   return VLimitName    ; break ;
	     case VCL:      return VCLName       ; break ;

	     case EName:    return ENameName     ; break ;
	     case EType:    return ETypeName     ; break ;
	     case ESymm:    return ESymmName     ; break ;
	     case EPlus:    return EPlusName     ; break ;
	     case EMinus:   return EMinusName    ; break ;

	     case QStatus:  return QStatusName   ; break ;
	     case QTitle:   return QTitleName    ; break ;
	     case QComment: return QCommentName  ; break ;
	     case QTag:     return QTagName      ; break ;

	     case HStatus:   return HStatusName   ; break ;
	     case HCategory: return HCategoryName ; break ;
	     case HName:     return HNameName     ; break ;
	     case HTitle:    return HTitleName    ; break ;

	     case OUri:      return OUriName       ; break ;
	     case OGroup:    return OGroupName     ; break ;
	     case OOwner:    return OOwnerName     ; break ;
	     case ODateTime: return ODateTimeName  ; break ;

	     case MSrc:     return MSrcName      ; break ;
	     case MProd:    return MProdName     ; break ;
	     case MQwn:     return MQwnName      ; break ;
	     case MTag:     return MTagName      ; break ;
     }
     return BlankName ;
}


Element::Type Element::elementType( const unsigned char* localName, Element::Type regn )
{
   // element names are always ascii so presumably casting away "unsigned" is OK
   const char* s = (const char*)localName ; 

   if( regn == Undefined ){  // Undefined allows checking for container elem, to allows regn switching    

        if(          strcmp(s, CQuoteName ) == 0){ return CQuote ; }
	else if(     strcmp(s, CModeName )  == 0){ return CMode  ; }
	else if(     strcmp(s, CValueName ) == 0){ return CValue ; }
	else if(     strcmp(s, CErrName )   == 0){ return CErr   ; }
	else if(     strcmp(s, CXErrName )  == 0){ return CXErr  ; }
	else if(     strcmp(s, CHeaderName ) == 0){ return CHeader ; }
	else if(     strcmp(s, COriginName ) == 0){ return COrigin ; }
        return  Undefined ;

   } else if( regn == CQuote ){	   // quote has leaf elements as well as container elements

	if(     strcmp(s, QStatusName )   == 0){ return QStatus  ; }
        else if(strcmp(s, QTitleName )    == 0){ return QTitle   ; }
        else if(strcmp(s, QCommentName )  == 0){ return QComment ; }
        else if(strcmp(s, QTagName )      == 0){ return QTag     ; }

        else if(strcmp(s, CModeName )     == 0){ return CMode  ; }
        else if(strcmp(s, CValueName )    == 0){ return CValue ; }
        else if(strcmp(s, CErrName )      == 0){ return CErr   ; }
        else if(strcmp(s, CXErrName )     == 0){ return CXErr  ; }

        return  Undefined ;

   } else if( regn == CMode ){	   

        if(     strcmp(s, MSrcName )    == 0){ return MSrc    ; }
        else if(strcmp(s, MProdName )   == 0){ return MProd   ; }
        else if(strcmp(s, MQwnName )    == 0){ return MQwn    ; }
        else if(strcmp(s, MTagName )    == 0){ return MTag    ; }
        return  Undefined ;

   } else if( regn == CValue ){	   

        if(     strcmp(s, VNumberName )  == 0){ return VNumber  ; }
        else if(strcmp(s, VUnitName )    == 0){ return VUnit    ; }
        else if(strcmp(s, VLimitName )   == 0){ return VLimit   ; }
        else if(strcmp(s, VCLName )      == 0){ return VCL      ; }
        return Undefined ; 

   } else if( regn == CErr || regn == CXErr  ){	   

        if(     strcmp(s, ENameName )  == 0){ return EName    ; }
        else if(strcmp(s, ETypeName )  == 0){ return EType    ; }
        else if(strcmp(s, ESymmName )  == 0){ return ESymm    ; }
        else if(strcmp(s, EPlusName )  == 0){ return EPlus    ; }
        else if(strcmp(s, EMinusName ) == 0){ return EMinus   ; }
        return Undefined ; 

   } else if( regn == CHeader ){   // header has leaf elements as well as one container element

	if(      strcmp(s, COriginName )    == 0){ return COrigin    ; }

	else if( strcmp(s, HStatusName )    == 0){ return HStatus    ; }
	else if( strcmp(s, HCategoryName )  == 0){ return HCategory  ; }
	else if( strcmp(s, HNameName )      == 0){ return HName      ; }
	else if( strcmp(s, HTitleName )     == 0){ return HTitle     ; }
        return Undefined ; 

   } else if( regn == COrigin ){	 

        if(      strcmp(s, OUriName )      == 0){ return OUri      ; }
        if(      strcmp(s, OGroupName )    == 0){ return OGroup    ; }
	else if( strcmp(s, OOwnerName )    == 0){ return OOwner    ; }
	else if( strcmp(s, ODateTimeName ) == 0){ return ODateTime ; }
        return Undefined ; 

   } 
   return Undefined ; 
}


void Element::read_quote( Quote& quote , XmlValue& qval )
{
    // reading the quote into data structure
   
    XmlEventReader& qrdr = qval.asEventReader() ;
    read_quote_( quote , qrdr );
    qrdr.close();

    // now after quote specifics look into the quotes header and its place within its root
    // shimmy up to the rez	
    //
    const XmlDocument& rezd = qval.asDocument();
    XmlValue rezv(rezd);

    //clog << "Element::read_quote " << rezd << " " << rezd.getName() << endl ;	
    //clog << "Element::read rezv " << rezv.asString() << endl ;

    XmlEventReader& rezr = rezv.asEventReader() ;

    // shuffle down to the CHeader
    Type curr = Undefined ;
    while (rezr.hasNext() && curr != CHeader ) {
	XmlEventReader::XmlEventType type = rezr.next();
	if (type == XmlEventReader::StartElement) curr = elementType( rezr.getLocalName(), Undefined );
    }
    read_header_( quote._header, rezr );    // "fill" better that "read"


    // continue on down looking at QTags of quote and siblings  
    int ipos = 0 ;

    while (rezr.hasNext() ) {
	XmlEventReader::XmlEventType type = rezr.next();

	if (type == XmlEventReader::StartElement){

	    curr = elementType( rezr.getLocalName(), CQuote );

        } else if ( type == XmlEventReader::Characters && curr == QTag ){

            ++ipos ;
	    size_t len ;	
            const unsigned char* value = rezr.getValue(len);
            const char* blnk = "" ;
	    string qtag = (rezr.isWhiteSpace()) ? blnk : reinterpret_cast<const char*>(value)  ; 	    

	    const char* msg = "" ;
            if( qtag == quote._qtag ){
                if( quote._qpos == 0 ){		    
                    quote._qpos = ipos ;
		    msg = "** match **";
		} else {
                    quote._qpos = -999 ;
		    msg = "** ERROR multi-match **";    // how can I throw exxeptions from external functions ?
		}	
            } 		    
	    clog << "qpos " << ipos << " " << qtag << " " << msg << endl ;  

	} else if (type == XmlEventReader::EndElement){

	    curr = Undefined ;

	}
    }
 
    rezr.close();

}	


void Element::read_header_( Header& header , XmlEventReader& rdr )
{
    //clog << "Element::read_header_ " << endl ;	
    Type regn(Undefined);
    Type curr(Undefined);

    while (rdr.hasNext()) {
	XmlEventReader::XmlEventType type = rdr.next();
	if (type == XmlEventReader::StartElement) 
	{
            if (!rdr.isEmptyElement()){
		curr = elementType( rdr.getLocalName(), regn );
		if( curr == CHeader || curr == COrigin  ) regn = curr ; 
            }		    
        } 
	else if ( type == XmlEventReader::Characters )
	{
	    size_t len ;	
            const unsigned char* value = rdr.getValue(len);
            const char* blnk = "" ;
	    string text = (rdr.isWhiteSpace()) ? blnk : reinterpret_cast<const char*>(value)  ; 	    
            fillHeader(header, regn, curr, text );           // no need to go to string here ?
	}
	else if ( type == XmlEventReader::EndElement )
	{
	    // pass Undefined as looking for leaving regn 
	    curr = elementType( rdr.getLocalName(), Undefined ); 
            if( curr == CHeader ) return ;  // job done 
            if( curr == regn ) regn = CHeader ;   

	    // simple approach works due to the small depth of the hierarchy, 
	    // for going deeper need to hold parent types
	}
   }	

}

void Element::read_quote_(Quote& q, XmlEventReader& rdr )
{
   Type regn(Undefined);
   Type curr(Undefined);

   // traverse Xml taking note of leaf container elements "rez:mode" "rez:value" "rez:err" "rez:xerr"
   // and parsing text 

   while (rdr.hasNext()) {
	XmlEventReader::XmlEventType type = rdr.next();
	if (type == XmlEventReader::StartElement) 
	{
            if (!rdr.isEmptyElement()){
		curr = elementType( rdr.getLocalName(), regn );
		if( curr == CQuote || curr == CMode  || curr == CValue || curr == CErr || curr == CXErr ) regn = curr ; 
            }		    
        } 
	else if ( type == XmlEventReader::Characters )
	{
	    size_t len ;	
            const unsigned char* value = rdr.getValue(len);
            const char* blnk = "" ;
	    string text = (rdr.isWhiteSpace()) ? blnk : reinterpret_cast<const char*>(value)  ; 	    
            fillQuote(q, regn, curr, text );           // no need to go to string here ?
	}
	else if ( type == XmlEventReader::EndElement )
	{
	    // pass Undefined as looking for leaving regn 
	    curr = elementType( rdr.getLocalName(), Undefined ); 
            if( curr == regn ) regn = CQuote ;   
	    // simple approach works due to the small depth of the hierarchy, 
	    // for going deeper need to hold parent types
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

void Element::fillHeader( Header& h, Element::Type regn , Element::Type curr , string& text )
{
    if(regn == Undefined ) return ;
    //clog << "Element::fillHeader " << regn << "=" << elementName(regn) << " : " << curr << "=" << elementName(curr) << " : " << text << endl;

    const char* s = text.c_str(); 
    if(     regn == COrigin ){

        switch(curr)
        {
	    case OUri:      fillString(s, h._origin._uri)       ; break ;
	    case OGroup:    fillString(s, h._origin._group)     ; break ;
	    case OOwner:    fillString(s, h._origin._owner)     ; break ;
	    case ODateTime: fillString(s, h._origin._datetime)  ; break ;
        }

    } else if( regn == CHeader ){

        switch(curr)
        {
	    case HStatus:      fillString(s, h._status)      ; break ;
	    case HCategory:    fillString(s, h._category)    ; break ;
	    case HName:        fillString(s, h._name)        ; break ;
	    case HTitle:       fillString(s, h._title)       ; break ;
        }

    }	    
}




void Element::fillQuote( Quote& q, Element::Type regn , Element::Type curr , string& text )
{
    //clog << "Element::fillQuote " << regn << "=" << elementName(regn) << " : " << curr << "=" << elementName(curr) << " : " << text << endl;
    if(regn == Undefined ) return ;

    const char* s = text.c_str(); 
    if(     regn == CQuote ){

        switch(curr)
        {
	    case QStatus:     fillString(s, q._status)   ; break ;
	    case QTitle:      fillString(s, q._title)    ; break ;
	    case QComment:    fillString(s, q._comment)  ; break ;
	    case QTag:        fillString(s, q._qtag)     ; break ;
        }

    } else if(regn == CMode ) {

	vector<Factor>& facv = q._factor ;    
        if(curr == regn) facv.push_back( Factor() ) ;     
	Factor& fac = facv.back() ;    
        switch(curr)
	{
	    case MSrc:    fillInt(   s, fac._src)  ; break ;
	    case MProd:   fillIntv(  s, fac._prod) ; break ;
	    case MQwn:    fillString(s, fac._qwn)  ; break ;
	    case MTag:    fillString(s, fac._mtag) ; break ;
	}

    } else if(regn == CValue ) {	   

	Val& val = q._val ;    
        switch(curr)
	{
	    case VNumber:  fillDouble(s, val._number) ; break ;
	    case VUnit:    fillDouble(s, val._unit)   ; break ;
	    case VLimit:   fillDouble(s, val._limit)  ; break ;
	}	 

    } else if( regn == CErr || regn == CXErr ) {	    

        vector<Err>& errv = ( regn == CErr ) ? q._err : q._xerr ;  // refer to the appropriate vector of the 2
        if(curr == regn) errv.push_back( Err() ) ;                 // on entering "top-level" elements, mint a new Err 
	Err& err = errv.back();                                    // refer to last Err in the vector
        switch(curr)
	{
	    case ESymm:   fillDouble(s, err._symm    ) ; break ;
	    case EPlus:   fillDouble(s, err._plus    ) ; break ;
	    case EMinus:  fillDouble(s, err._minus   ) ; break ;
	    case EName:   fillString(s, err._errname ) ; break ;
	}

    }

}


