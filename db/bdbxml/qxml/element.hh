#ifndef ELEMENT_HH
#define ELEMENT_HH

#include <string>
#include <vector>
#include "dbxml/DbXml.hpp"

using namespace std ;
using namespace DbXml ;

class Quote ;
class Header ;

class Element
{
   public:	
          enum Type {
	          Undefined,
                  //
                  CHeader,
                  COrigin,
                  CQuote,
		  CMode,
                  CValue,
                  CErr,
                  CXErr,
                  //
		  HStatus,
		  HCategory,
		  HName,
		  HTitle,
		  //
		  OUri,
		  OGroup,
		  OOwner,
		  ODateTime,
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
	          ESymm,
	          EPlus,
	          EMinus,
		  //
		  Last
	  };   
          // approach relies on global distinction between element names

	  static const char* BlankName;

 	  static const char* CHeaderName;
 	  static const char* COriginName;
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

  	  static const char* HStatusName;
	  static const char* HCategoryName ;
	  static const char* HNameName ;
	  static const char* HTitleName ;

  	  static const char* OUriName;
	  static const char* OGroupName ;
	  static const char* OOwnerName ;
	  static const char* ODateTimeName ;

  	  static const char* ENameName;
	  static const char* ETypeName ;
	  static const char* ESymmName ;
	  static const char* EPlusName ;
	  static const char* EMinusName ;

          static Element::Type elementType(const unsigned char* localName, Element::Type regn );
          const char* elementName( Element::Type type );

	  Element(){};
          virtual ~Element(){};

	  void read_quote(   Quote& quote, XmlValue& val );
	  void read_quote_(  Quote& quote, XmlEventReader& rdr );

	  void read_header(   Header& header, XmlValue& val );
	  void read_header_(  Header& header, XmlEventReader& rdr );

          void fillQuote(  Quote& q , Element::Type posn , Element::Type curr , string& text );
          void fillHeader( Header& h, Element::Type posn , Element::Type curr , string& text );

  	  void fillDouble( const char* s , double& out );
	  void fillString( const char* s , string& out );
          void fillInt(    const char* s , int& out );
          void fillIntv(   const char* s , vector<int>& out );

   private:

};

#endif
