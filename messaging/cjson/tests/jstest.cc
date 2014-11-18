#include "js.hh"

#include <cassert>

int main(int argc, char** argv)
{
    assert(argc>1);
    JS* js = JS::Load(argv[1]);
    js->Print();

    const char* wanted = argc > 2 ? argv[2] : "" ;
    js->Traverse(wanted);

    delete js;
}


