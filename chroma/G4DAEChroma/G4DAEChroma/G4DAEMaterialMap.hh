#ifndef G4DAEMATERIALMAP_H
#define G4DAEMATERIALMAP_H 1

#include "G4Material.hh"
#include "G4DAEChroma/G4DAEMap.hh"
#include <string>
#include <map>

class G4DAEMetadata ; 
class G4DAEMaterialMap ; 

class G4DAEMaterialMap
{
public:
    static int* MakeLookupArray(G4DAEMaterialMap* geant4, G4DAEMaterialMap* chroma);
    static void DumpLookupArray( G4DAEMaterialMap* a, G4DAEMaterialMap* b, int* a2b );

    G4DAEMaterialMap(const G4MaterialTable* table=NULL);
    G4DAEMaterialMap(G4DAEMetadata* meta, const char* key );
    virtual ~G4DAEMaterialMap();
    static void MakeGeant4MaterialMap();
    void Print(const char* msg="G4DAEMaterialMap::Print");
    int GetMaxIndex();
    int GetMinIndex();
    int FindIndex(const char* name);
    std::string FindName(int index);
    std::string GetKey();

    Map_t GetStringMap();

private:
    std::string m_key ; 
    std::map<std::string, int> m_map ; 


};

#endif
