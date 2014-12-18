// cjs-;cjs-jstest-make    # Build and run 

#include "cJSON/js.hh"
#include <string>
#include <cassert>

int main(int argc, char** argv)
{
    assert(argc>1);
    JS* js = JS::Load(argv[1]);

    Map_t map ; 
    // without COLUMNS type specifiers JS excludes from returned maps
    map["COLUMNS"] = "hello:s,world:s,fnum:f,inum:i" ;  
    map["hello"] = "world" ;
    map["world"] = "hello" ;
    map["fnum"] = "1.1" ;
    map["inum"] = "101" ;

    js->AddMap("extra", map);

    js->SetKV("parameters", "deviceid", "10");

    js->Print();

    const char* wanted = argc > 2 ? argv[2] : "" ;
    js->Traverse(wanted);



    const char* paths[] = {"/results", "/extra", NULL };
    for(int i=0 ; i < 3 ; i++ )
    {
       Map_t m = js->GetMap(paths[i]);
       JS::DumpMap(m, paths[i] );
    }



    const char* rawpath = "/chroma_material_map" ;
    Map_t raw = js->GetRawMap(rawpath);
    JS::DumpMap(raw, rawpath);



    delete js;
}


