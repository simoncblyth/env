#include <iostream>
#include <cpr/cpr.h>

int main()
{
    cpr::Response r = cpr::Get(cpr::Url{"https://api.github.com/repos/whoshuu/cpr/contributors"},
                      cpr::Authentication{"user", "pass", cpr::AuthMode::BASIC},
                      cpr::Parameters{{"anon", "true"}, {"key", "value"}});

    std::cout 
       << "r.status_code " << r.status_code << "\n"
       << "r.header[\"content-type\"] [" << r.header["content-type"] << "]\n"
       << "[r.text\n" 
       <<  r.text
       << "]r.text\n" 
       ;

    return 0 ; 
}
