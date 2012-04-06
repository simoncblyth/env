#ifndef ELEMENT_HH
#define ELEMENT_HH

#include <string>
#include <vector>
#include "dbxml/DbXml.hpp"

using namespace std ;
using namespace DbXml ;

class Quote ;

class Element
{
   public:	
          enum Type {
	          Undefined,
                  //
                  CQuote,
		  CMode,
                  CValue,
                  CErr,
                  CXErr,
                  //
                  QStatus,
                  QTitle,
                  QComment,
                  QTag,
                  //
                  MQwn,
                  MSrc,
                  MProd,
                  MTag,
		  //
	          VNumber,
	          VUnit,
	          VLimit,
	          VCL,
		  //
	          EName,
	          EType,
	          EPlus,
	          EMinus,
		  //
		  Last
	  };   
          // approach relies on global distinction between element names

	  static const char* BlankName;

 	  static const char* CQuoteName;
 	  static const char* CModeName;
 	  static const char* CValueName;
	  static const char* CErrName ;
	  static const char* CXErrName ;

 	  static const char* QStatusName;
 	  static const char* QTitleName;
 	  static const char* QCommentName;
 	  static const char* QTagName;

	  static const char* MQwnName ;
	  static const char* MSrcName ;
	  static const char* MProdName ;
	  static const char* MTagName ;

  	  static const char* VNumberName;
	  static const char* VUnitName ;
	  static const char* VLimitName ;
	  static const char* VCLName ;

  	  static const char* ENameName;
	  static const char* ETypeName ;
	  static const char* EPlusName ;
	  static const char* EMinusName ;

          static Element::Type elementType(const unsigned char* localName, Element::Type regn );
          const char* elementName( Element::Type type );

	  Element(){};
          virtual ~Element(){};

	  void read(   Quote& q, XmlValue& val );
	  void read_(  Quote& q, XmlEventReader& rdr );
          void parse( Quote& q, Element::Type posn , Element::Type curr , string& text );

  	  void fillDouble( const char* s , double& out );
	  void fillString( const char* s , string& out );
          void fillInt( const char* s , int& out );
          void fillIntv( const char* s , vector<int>& out );

   private:

};

#endif
