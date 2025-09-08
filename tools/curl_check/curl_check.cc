/**

https://gist.github.com/whoshuu/2dc858b8730079602044
**/


#include "curl/curl.h"
#include <string>


#include <iostream>
#include <iomanip>


size_t writeFunction(void *ptr, size_t size, size_t nmemb, std::string* data) 
{
    data->append((char*) ptr, size * nmemb);
    return size * nmemb;
}


int main()
{
    auto curl = curl_easy_init();
    if (!curl) return 1 ;

    curl_easy_setopt(curl, CURLOPT_URL, "https://api.github.com/repos/whoshuu/cpr/contributors?anon=true&key=value");
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl, CURLOPT_USERPWD, "user:pass");
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "curl/7.42.0");
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 50L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);

    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // follow redirects

    
    std::string response_string;
    std::string header_string;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeFunction);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &header_string);
    
    char* url;
    long response_code;
    double elapsed;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME, &elapsed);
    curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &url);
    
    curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    curl = NULL;

    std::cout 
        << "url [" << ( url ? url : "-" ) << "]\n"
        << "response_code " << response_code << "\n"
        << "elapsed " << std::fixed << std::setw(10) << std::setprecision(3) << elapsed << "\n"
        << " [response_string\n" 
        << response_string
        << " ]response_string\n" 
        << " [header_string\n" 
        << header_string
        << " ]header_string\n" 
        ;



    return 0 ; 
}
