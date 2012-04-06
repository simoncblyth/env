#include "model.hh"

#include <iostream>

typedef vector<Factor> facv ;
typedef vector<Err> errv ;
typedef vector<int> intv ;



void Header::dump() const 
{
    clog << "Header " << _name << "," << _status << "," << _category << endl ;
    clog << "   " << _title << endl ;
    _origin.dump();
}


void Origin::dump() const 
{
    clog << "Origin " << _uri << "," << _group << "," << _owner << "," << _datetime << endl ;
}


void Factor::dump() const 
{
    clog << "Factor " << _qwn << "[" << _src ; 
    if( _prod.size() > 0){
	 clog << " => ( ";
         intv::const_iterator ip ;
         for( ip = _prod.begin() ; ip != _prod.end() ; ++ip ) clog << *ip << " " ;
         clog << " ) ";
    }	 
    clog << "] " << _mtag << endl ;
}

void Quote::dump() const
{
   clog << "Quote " << endl ;
   clog << "    qpos: " << _qpos << " (1-based sibling position) " << endl ;
   clog << "    qtag: " << _qtag    << endl ;
   clog << "   title: " << _title   << endl ;
   clog << "  status: " << _status  << endl ;
   clog << " comment: " << _comment << endl ;

   _header.dump(); 

   facv::const_iterator ic ;
   for( ic = _factor.begin() ; ic != _factor.end() ; ++ic ) ic->dump();

   _val.dump();
   errv::const_iterator it ;
   clog << "err" << endl ; 
   for( it = _err.begin() ; it != _err.end() ; ++it ) it->dump();
   clog << "xerr" << endl ; 
   for( it = _xerr.begin() ; it != _xerr.end() ; ++it ) it->dump();
}

void Err::dump() const
{
   clog << "Err " 
	<< " errname " << _errname
	<< " symm "   << _symm 
	<< " plus "   << _plus 
	<< " minus "  << _minus 
	<< endl ; 
}


void Val::dump() const
{
   clog << "Val " 
	<< " number " << _number 
	<< " unit "   << _unit 
	<< " limit "  << _limit 
	<< endl ; 
}


