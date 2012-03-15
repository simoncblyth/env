#include <map>
#include <string>

using namespace std ;

typedef map<string, string> ssmap ;
typedef map<string, ssmap> sssmap ;

int qxml_config(int argc, char **argv, sssmap& m);

