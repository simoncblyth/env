// testing config handling 
#include "config.hh"
#include <iostream>
#include <boost/algorithm/string.hpp>

int main(int argc, char **argv)
{
    sssmap cfg ;
    int rc = qxml_config( argc, argv, cfg );

    string xqmpath = cfg["dbxml"]["dbxml.xqmpath"] ;
    cout << xqmpath << endl ;

    svec dirs;
    boost::split(dirs, xqmpath, boost::is_any_of(":"));
    svec::const_iterator it ;
    for( it = dirs.begin() ; it != dirs.end() ; ++it ){
        cout << *it << endl ;   
    }

    return rc ;
}
