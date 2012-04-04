#ifndef QUOTE_HH
#define QUOTE_HH

#include <string>
#include <vector>

#include "dbxml/DbXml.hpp"

using namespace std ;
using namespace DbXml ;


class Err
{
   public:
	  Err() : 
	      _errname(""),	  
	      _symm(0.),
	      _minus(0.),
	      _plus(0.) {} ;

          virtual ~Err(){};
   private:
	  string _errname ;  
          double _symm ;
	  double _plus ;
	  double _minus ;
};


class Val
{
   public:
	  Val() : 
	       _number(0.),
               _unit(0.),
               _limit(0.) {} ;	       
          virtual ~Val(){};
   private:
          double _number ;
	  double _unit ;
	  double _limit ;
};


class Quote
{
public:
	Quote(XmlEventReader& rdr);
	virtual ~Quote(){};
private:
        void parse( string& posn , string& curr , string& text );

        Val _val ;
        vector<Err> _err ;
        vector<Err> _xerr ;

};


#endif
