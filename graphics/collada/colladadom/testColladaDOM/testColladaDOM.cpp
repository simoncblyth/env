
#include <dae.h>
#include <dom/domCOLLADA.h>
#include <iostream>
#include <vector>

using namespace std ; 

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


int main(int argc, char * const argv[]){
    DAE dae(NULL,NULL,"1.4.1");  

     // nope cannot do this outside 
    //daeIOPlugin* plugin = (daeIOPlugin*)new daeLIBXMLPlugin(dae);
    //plugin->setMeta(
    //dae.setIOPlugin(plugin);

    //DAE dae(NULL,NULL,"1.5.0");  
    //const char* path = "file:///usr/local/env/geant4/geometry/daeserver/0___2.dae";
    //const char* path = "/usr/local/env/geant4/geometry/daeserver/0___2.dae";
    //daeInt error = dae.load(path);

    if( argc < 2 ){
        std::cout << "no path argument " << std::endl ; 
        return 1 ; 
    } 

    const char* path = argv[1] ; 
    ColladaDOM141::domCOLLADA* dom = (ColladaDOM141::domCOLLADA*)dae.open(path);

    cout << "dom: " << dom << endl ; 


    vector<daeElement*> nodes = dae.getDatabase()->typeLookup(domNode::ID());
    for (size_t i = 0; i < nodes.size(); i++) {
        daeElementRefArray children = nodes[i]->getChildren();
        for (size_t j = 0; j < children.getCount(); j++) {

            string sid = children[j]->getAttribute("sid");
            if (!sid.empty()) {
                 cout << "sid " << sid << endl ; 
            }

            string id = children[j]->getAttribute("id");
            if (!id.empty()) {
                 cout << "id " << id << endl ; 
            }


        }
    }    





    return 0;
}
