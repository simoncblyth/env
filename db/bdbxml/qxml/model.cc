#include "model.hh"

#include <iostream>

typedef vector<Factor> facv ;
typedef vector<Err> errv ;
typedef vector<int> intv ;

void Factor::dump() const 
{
    cout << "Factor " << _qwn << "[" << _src ; 
    if( _prod.size() > 0){
	 cout << " => ( ";
         intv::const_iterator ip ;
         for( ip = _prod.begin() ; ip != _prod.end() ; ++ip ) cout << *ip << " " ;
         cout << " ) ";
    }	 
    cout << "]" << endl ;
}

void Quo::dump() const
{
   cout << "Quo " << endl ;
   facv::const_iterator ic ;
   for( ic = _factor.begin() ; ic != _factor.end() ; ++ic ) ic->dump();

   _val.dump();
   errv::const_iterator it ;
   cout << "err" << endl ; 
   for( it = _err.begin() ; it != _err.end() ; ++it ) it->dump();
   cout << "xerr" << endl ; 
   for( it = _xerr.begin() ; it != _xerr.end() ; ++it ) it->dump();
}

void Err::dump() const
{
   cout << "Err " 
	<< " errname " << _errname
	<< " symm "   << _symm 
	<< " plus "   << _plus 
	<< " minus "  << _minus 
	<< endl ; 
}


void Val::dump() const
{
   cout << "Val " 
	<< " number " << _number 
	<< " unit "   << _unit 
	<< " limit "  << _limit 
	<< endl ; 
}


