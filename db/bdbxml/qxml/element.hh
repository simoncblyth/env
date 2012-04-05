#ifndef ELEMENT_HH
#define ELEMENT_HH

#include <string>
#include <vector>
#include "dbxml/DbXml.hpp"

using namespace std ;
using namespace DbXml ;

class Quo ;

class Element
{
   public:	
          enum Type {
	          Undefined,
                  //
                  Quote,
		  Mode,
                  Value,
                  Err,
                  XErr,
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

 	  static const char* QuoteName;
 	  static const char* ModeName;
 	  static const char* ValueName;
	  static const char* ErrName ;
	  static const char* XErrName ;

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

	  Element(){};
          virtual ~Element(){};

	  void read(XmlEventReader& rdr, Quo& q );
          void parse( Quo& q, Element::Type posn , Element::Type curr , string& text );
  	  void fillDouble( const char* s , double& out );
	  void fillString( const char* s , string& out );
          void fillInt( const char* s , int& out );
          void fillIntv( const char* s , vector<int>& out );

   private:

};

#endif
