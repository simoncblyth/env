#include "cJSON/js.hh"

#include <cassert>

int main(int argc, char** argv)
{
    assert(argc>1);
    JS* js = JS::Load(argv[1]);

    Map_t map ; 
    map["hello"] = "world" ;
    map["world"] = "hello" ;
    map["fnum"] = "1.1" ;
    map["inum"] = "101" ;

    js->AddMap("extra", map);


    js->Print();

    const char* wanted = argc > 2 ? argv[2] : "" ;
    js->Traverse(wanted);


    delete js;
}


