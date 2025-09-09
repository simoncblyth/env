#pragma once

#include <iomanip>
#include <sstream>

#include <cassert>
#include <string.h>

#include <curl/curl.h>

#include "NP_CURL_HDR.h"
#include "NP_CURL_Upload.h"
#include "NP_CURL_Download.h"


template<typename T>
struct NP_CURL
{
    const char* url ; 
    CURL* session;
    NP_CURL_Upload* upload ;

    NP_CURL_Download* download ;
    NP_CURL_HDR dhdr ; 


    CURLcode curl_code ;
    long     http_code ; 
    int      rc ; 

    struct curl_slist* headerlist ;


    NP_CURL(const char* url);
    virtual ~NP_CURL();

    void prepare_upload( const std::vector<T>& up );
    void prepare_download();
    int perform();
    int collect_download( std::vector<T>& down );
    std::string desc() const ;

};



template<typename T>
inline NP_CURL<T>::NP_CURL(const char* url_)
    :
    url( url_ ? strdup(url_) : nullptr ),
    session(nullptr),
    upload(new NP_CURL_Upload),
    download(new NP_CURL_Download),
    dhdr("down"),
    curl_code((CURLcode)0),
    http_code(0),
    rc(0),
    headerlist(nullptr)
{
    curl_global_init(CURL_GLOBAL_ALL);
    session = curl_easy_init();
}

template<typename T>
inline NP_CURL<T>::~NP_CURL()
{
    curl_easy_cleanup(session);
    curl_slist_free_all(headerlist);
    curl_global_cleanup();
}



template<typename T>
inline void NP_CURL<T>::prepare_upload( const std::vector<T>& up )
{
    std::cout << "[NP_CURL::prepare_upload\n" ;

    upload->size = up.size()*sizeof(T); 
    upload->data = (const char*)up.data() ;

    std::string token = "secret" ;
    std::string level = "1" ;
    std::string dtype = "float32" ;
    std::string shape = std::to_string(up.size()); 

    bool expected_dtype = NP_CURL_HDR::Expected_DTYPE(dtype.c_str());
    assert( expected_dtype );

    std::string x_token = NP_CURL_HDR::Format_TOKEN(token.c_str()) ;
    std::string x_level = NP_CURL_HDR::Format_LEVEL(level.c_str()) ;
    std::string x_dtype = NP_CURL_HDR::Format_DTYPE(dtype.c_str()) ;
    std::string x_shape = NP_CURL_HDR::Format_SHAPE(shape.c_str()) ;

    headerlist = curl_slist_append(headerlist, x_token.c_str() );
    headerlist = curl_slist_append(headerlist, x_level.c_str() );
    headerlist = curl_slist_append(headerlist, x_dtype.c_str() );
    headerlist = curl_slist_append(headerlist, x_shape.c_str() );

    curl_easy_setopt(session, CURLOPT_URL, url );
    curl_easy_setopt(session, CURLOPT_POST, 1L);
    curl_easy_setopt(session, CURLOPT_HTTPHEADER, headerlist);

    curl_easy_setopt(session, CURLOPT_READFUNCTION, NP_CURL_Upload::read_callback);
    curl_easy_setopt(session, CURLOPT_READDATA, upload );
    curl_easy_setopt(session, CURLOPT_POSTFIELDSIZE, (long)upload->size);

    std::cout << "]NP_CURL::prepare_upload\n" ;
}


template<typename T>
inline void NP_CURL<T>::prepare_download()
{
    std::cout << "[NP_CURL::prepare_download\n" ;

    download->buffer = (char*)malloc(1); // Start with a 1-byte buffer
    download->size = 0 ;
    download->buffer[0] = '\0';

    curl_easy_setopt(session, CURLOPT_WRITEFUNCTION, NP_CURL_Download::write_callback);
    curl_easy_setopt(session, CURLOPT_WRITEDATA, download );

    std::cout << "[NP_CURL::prepare_download\n" ;
}


template<typename T>
inline int NP_CURL<T>::perform()
{
    std::cout << "[NP_CURL::perform\n" ;
    curl_code = curl_easy_perform(session);
    if (curl_code != CURLE_OK)
    {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(curl_code));
        rc = 1 ; 
        return rc ; 
    }

    curl_easy_getinfo(session, CURLINFO_RESPONSE_CODE, &http_code);
    std::cout << "-NP_CURL::perform http_code[" << http_code << "]\n" ;

    if( http_code >= 400 ) 
    {
        std::cout 
           << "-NP_CURL::perform download.buffer[\n" 
           << download->buffer
           << "\n]\n"
           ;         
    }


    struct curl_header* h;
    struct curl_header* p = nullptr ;
    do 
    {
        h = curl_easy_nextheader(session, CURLH_HEADER, -1, p );
        if(h) dhdr.collect(h->name, h->value);
        p = h;
    } 
    while(h);

    dhdr.collect_json_content(download->buffer, download->size );


    std::cout << "]NP_CURL::perform\n" ;
    return rc ; 
}





template<typename T>
inline int NP_CURL<T>::collect_download( std::vector<T>& b )
{
    std::cout << "[NP_CURL::collect_download\n" ;
    rc = perform();
    if( rc != 0 ) return rc ; 

    if( http_code != 200 ) 
    {
        fprintf(stderr, "collect_download abort:  http_code %ld\n", http_code);
        rc = 2 ;
        return rc ; 
    } 


    int num = dhdr.sh.size() == 1 ? dhdr.sh[0] : -1 ;  
    if(num == -1) rc = 2 ; 

    if(num > 0)
    {
        b.resize(num);
        const T* tbuf = (T*)download->buffer ; 

        T* bb = b.data();
        for(int i=0 ; i < num ; i++ ) bb[i] = tbuf[i] ; 
     }

    std::cout << "]NP_CURL::collect_download\n" ;

    return rc ; 
}


template<typename T>
inline std::string NP_CURL<T>::desc() const 
{
    std::stringstream ss ; 
    ss 
       << "[NP_CURL::desc\n"
       << download->desc() << "\n"
       << dhdr.desc() << "\n"
       << "]NP_CURL::desc\n"
       ;

    std::string str = ss.str() ;
    return str ;  
}
 


