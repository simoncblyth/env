#ifndef QUOTE_HH
#define QUOTE_HH

#include <string>
#include <vector>

using namespace std ;


class Origin
{
   public:
	  Origin() :
                   _uri(""),
		   _group(""),
		   _owner(""),
		   _datetime("") {};

	  virtual ~Origin(){};  
          void dump() const;

          string _uri ;
          string _group ;
          string _owner ;
          string _datetime ;
};



class Header
{
   public:
	  Header() :
                   _status(""),
		   _category(""),
		   _name(""),
		   _title("") {};

	  virtual ~Header(){};  
          void dump() const;

          string _status ;
          string _category ;
          string _name ;
          string _title ;
	  Origin _origin ; 
};

class Factor
{
   public:
	  Factor() : _src(0),
	             _qwn(""),
		     _mtag("") {};

          virtual ~Factor(){};
          void dump() const;

	  int _src ;
	  vector<int> _prod ;
	  string _qwn ;
	  string _mtag ;
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


class Quote
{
   public:
	Quote() : 
	      _status(""),
	      _title(""),
	      _comment(""),
	      _qtag(""),
              _qpos(0) {} ;

	virtual ~Quote(){};
        void dump() const;


	Header _header ;
        Val _val ;
	vector<Factor> _factor ;
        vector<Err> _err ;
        vector<Err> _xerr ;

        string _status ;
	string _title ;
	string _comment ;
	string _qtag ;

	int _qpos ;   // 1-based sibling position of the quote within its rez:rez

  private:

};


#endif
