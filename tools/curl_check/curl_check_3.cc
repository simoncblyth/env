/**
curl_check_3.cc
=================

CHECK=3 ~/e/tools/curl_check/curl_check.sh

This is limited to 1D arrays .. no point doing that when NP.hh 
has all that functionality already.  

**/


#include <vector>
#include <iostream>

#include "NP_CURL.h"


int main(void) 
{
    const char* endpoint = "http://127.0.0.1:8000/array_transform" ;

    std::vector<float> a = { 0.f, 1.f, 2.f, 3.f };
    std::vector<float> b ; 


    NP_CURL<float> nc(endpoint) ; 
    nc.prepare_upload( a );
    nc.prepare_download();
    nc.collect_download(b);

    std::cout << nc.desc() ;  

    for(int i=0 ; i < int(b.size()) ; i++ ) std::cout << b[i] << "\n" ;


    return 0;
}



