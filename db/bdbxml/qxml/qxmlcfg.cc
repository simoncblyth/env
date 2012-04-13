// testing config handling 
#include "config.hh"
#include "potools.hh"
#include <iostream>
#include <boost/algorithm/string.hpp>

int main(int argc, char **argv)
{
    sssmap cfg ;
    int rc = qxml_config( argc, argv, cfg , true );

    string level = cfg["cli"]["level"] ;
    cout << level << endl ;
    string xqmpath = cfg["dbxml"]["dbxml.xqmpath"] ;
    cout << xqmpath << endl ;

    svec dirs;
    boost::split(dirs, xqmpath, boost::is_any_of(":"));
    svec::const_iterator it ;
    for( it = dirs.begin() ; it != dirs.end() ; ++it ){
        cout << *it << endl ;   
    }


    cout << "cfg_dump " << endl ; 
    cfg_dump( cfg );

    return rc ;
}
