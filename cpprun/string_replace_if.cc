/*

*/
#include <string>
#include <algorithm>
#include <iostream>

using namespace std ;

bool colon_or_slash(const char c )
{
    switch (c)
    {
        case ':':
        case '/':
        return true;
    default:
        return false;
    }   
}


int main()
{
    string s = "Hello:World:Hello/Hello/Hello/Hello" ;
    cout << s << '\n';
 
    //replace(s.begin(), s.end(), ":", "_");  this doesnt compile, must be single quotes to demote characters
    replace(s.begin(), s.end(), ':', '_');
    cout << s << '\n';

    replace_if(s.begin(), s.end(), &colon_or_slash, '_');
    cout << s << '\n';

    return 0 ;
}


