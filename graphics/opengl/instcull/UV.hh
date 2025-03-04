#pragma once 
#include <string>
#include "DEMO_API_EXPORT.hh"
    
struct DEMO_API UV  
{                 
    unsigned s() const {   return s_ & 0xffff ; } 
    unsigned u() const {   return u_ & 0xffff ; }
    unsigned v() const {   return v_ & 0xffff ; }
    unsigned p() const {   return p_ ; }   // primitive index
    unsigned nu() const {  return u_ >> 16 ; }
    unsigned nv() const {  return v_ >> 16 ; } ;         
    unsigned ps() const {  return p()*100 + s() ; } 
                  
    static unsigned ps_to_prim( unsigned ps_){ return ps_/100   ; }
    static unsigned ps_to_sheet(unsigned ps_){ return ps_ % 100 ; }
               
    bool matches(unsigned prim, unsigned sheet) const { return prim == p() && sheet == s() ; }
                 
    float   fu() const  { return float(u())/float(nu()) ; }
    float   fv() const  { return float(v())/float(nv()) ; }
        
    float  fu2pi(bool seamed=true) const ;
    float  fv2pi(bool seamed=true) const ;  
    float  fvpi() const ; 
               
    bool is_interior(unsigned margin=0) const ;

    std::string desc() const ;
    std::string detail() const ;
                  
    unsigned s_ ; 
    unsigned u_ ; 
    unsigned v_ ;
    unsigned p_ ;
};      
                  
                  
                  
inline UV make_UV(unsigned s, unsigned u, unsigned v, unsigned nu, unsigned nv, unsigned p)
{              
   UV uv ;

   uv.s_ = s ;
   uv.u_ = (nu << 16) | u  ;
   uv.v_ = (nv << 16) | v  ;
   uv.p_ = p ; 

   return uv ;
}

