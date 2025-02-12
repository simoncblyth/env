// ggv --lookup
// ggv --jpmt --lookup


#include "Opticks.hh"

#include "GBndLib.hh"
#include "Lookup.hpp"

#include "GGEO_BODY.hh"
#include "PLOG.hh"



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    Opticks* opticks = new Opticks(argc, argv);

    GBndLib* blib = GBndLib::load(opticks, true );

    blib->dump();



    Lookup* m_lookup = new Lookup();

    m_lookup->loadA( opticks->getIdFold(), "ChromaMaterialMap.json", "/dd/Materials/") ;

    blib->fillMaterialLineMap( m_lookup->getB() ) ;    

    m_lookup->crossReference();

    m_lookup->dump("ggeo-/LookupTest");




    printf("  a => b \n");
    for(unsigned int a=0; a < 35 ; a++ )
    {   
        int b = m_lookup->a2b(a);
        std::string aname = m_lookup->acode2name(a) ;
        std::string bname = m_lookup->bcode2name(b) ;
        printf("  %3u -> %3d  ", a, b );

        if(b < 0) printf(" %25s : WARNING failed to translate acode %u \n", aname.c_str(), a);    
        else
        {   
             assert(aname == bname);
             printf(" %25s \n", aname.c_str() );
        }   
    }   

    


}
