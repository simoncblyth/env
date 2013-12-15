
#include <dae.h>
#include <dom/domCOLLADA.h>
#include <iostream>

void write_minimal_valid()
{
    DAE dae;
    daeElement* root = dae.add("valid.dae");
    daeElement* asset = root->add("asset");
    daeElement* contributor = asset->add("contributor");
    daeElement* created = asset->add("created");
    daeElement* modified = asset->add("modified");
    const char* date = "2008-04-08T13:07:52-08:00";
    created->setCharData(date);
    modified->setCharData(date);
    dae.writeAll();
}



int main() {
    DAE dae(NULL,NULL,"1.4.1");  

     // nope cannot do this outside 
    //daeIOPlugin* plugin = (daeIOPlugin*)new daeLIBXMLPlugin(dae);
    //plugin->setMeta(
    //dae.setIOPlugin(plugin);

    //DAE dae(NULL,NULL,"1.5.0");  
    //const char* path = "file:///usr/local/env/geant4/geometry/daeserver/0___2.dae";
    const char* path = "/usr/local/env/geant4/geometry/daeserver/0___2.dae";
    //daeInt error = dae.load(path);

    ColladaDOM141::domCOLLADA* dom = (ColladaDOM141::domCOLLADA*)dae.open(path);

    std::cout << "dom: " << dom << std::endl ; 


    return 0;
}
