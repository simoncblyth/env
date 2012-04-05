#ifndef QUOTE_HH
#define QUOTE_HH

#include <string>
#include <vector>

using namespace std ;


class Factor
{
   public:
	  Factor() : _src(0),
	             _qwn(""),
		     _tag("") {};

          virtual ~Factor(){};
          void dump() const;

	  int _src ;
	  vector<int> _prod ;
	  string _qwn ;
	  string _tag ;
};	

class Err
{
   public:
    	  Err() : 
	      _errname(""),	  
	      _symm(0.),
	      _minus(0.),
	      _plus(0.) {} ;

          virtual ~Err(){};
          void dump() const;

	  string _errname ;  
          double _symm ;
	  double _plus ;
	  double _minus ;
   private:
};

class Val
{
   public:
	  Val() : 
	       _number(0.),
               _unit(0.),
               _limit(0.) {} ;	       

          virtual ~Val(){};
          void dump() const;

          double _number ;
	  double _unit ;
	  double _limit ;
   private:
};


class Quo
{
   public:
	Quo() : 
	      _status(""),
	      _title(""),
	      _comment(""),
	      _qtag("") {} ;

	virtual ~Quo(){};
        void dump() const;

        Val _val ;
	vector<Factor> _factor ;
        vector<Err> _err ;
        vector<Err> _xerr ;

        string _status ;
	string _title ;
	string _comment ;
	string _qtag ;

  private:

};


#endif
