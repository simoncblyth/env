// see  instcull-vi for notes
#include "Geom.hh"
#include "Demo.hh"

int main()
{
    Geom geom(3,300) ; 

    Demo app(&geom) ; 
    app.renderLoop();      
    exit(EXIT_SUCCESS);
}


