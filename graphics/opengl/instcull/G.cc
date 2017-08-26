#include <iostream>
#include <sstream>
#include <iomanip>

#include "G.hh"


std::string G::gpresent( const char* label, const glm::mat4& m, unsigned prec, unsigned wid, unsigned lwid, unsigned mwid, bool flip  )
{
    std::stringstream ss ; 
    for(int i=0 ; i < 4 ; i++)
    {   
        ss << std::setw(lwid) << ( i == 0 ? label : " " ) << std::setw(mwid) << " m[" << i << "] = { " ; 
        for(int j=0 ; j < 4 ; j++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) << ( flip ? m[j][i] : m[i][j] )  << ", " ; 
        ss << " } ; " << std::endl ; 
    }      
    return ss.str();
}

std::string G::gpresent( const char* label, const glm::vec4& m, unsigned prec, unsigned wid, unsigned lwid, unsigned mwid  )
{
    std::stringstream ss ;   
    ss << std::setw(lwid) << label << std::setw(mwid) << " " ;
    for(int j=0 ; j < 4 ; j++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) <<  m[j]  << " " ; 
    return ss.str();
}

std::string G::gpresent( const char* label, const glm::vec3& m, unsigned prec, unsigned wid, unsigned lwid, unsigned mwid  )
{
    std::stringstream ss ;   
    ss << std::setw(lwid) << label << std::setw(mwid) << " " ;
    for(int j=0 ; j < 3 ; j++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) <<  m[j]  << " " ; 
    return ss.str();
}

std::string G::gpresent( const char* label, const glm::uvec4& m, unsigned prec, unsigned wid, unsigned lwid, unsigned mwid  )
{
    std::stringstream ss ;   
    ss << std::setw(lwid) << label << std::setw(mwid) << " " ;
    for(int j=0 ; j < 4 ; j++) ss << std::setprecision(prec) << std::fixed << std::setw(wid) <<  m[j]  << " " ; 
    return ss.str();
}

