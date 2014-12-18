#include "G4DAEChroma/G4DAEMaterialMap.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEMap.hh"

#include "G4Material.hh"

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdio.h>

using namespace std ; 

typedef map<string,int> SI_t ;


G4DAEMaterialMap::G4DAEMaterialMap(G4DAEMetadata* meta, const char* key )
{
    if(!meta) return ;

    m_key = key ; 
    Map_t raw = meta->GetRawMap(key);

    if(raw.size() == 0) 
    {
        printf("G4DAEMaterialMap key %s is missing \n", key );
        return ;
    }

    for(Map_t::iterator it=raw.begin(); it != raw.end() ; it++)
    {
        string key = it->first ; 
        string val = it->second ; 
        int ival ;
        istringstream(val) >> ival ; 
        m_map[key] = ival ; 
        //printf("key %40s  val %5s   ival %d \n", key.c_str(), val.c_str(), ival  );    
    }
}

G4DAEMaterialMap::G4DAEMaterialMap(const G4MaterialTable* table)
{
    m_key = "G4MaterialTable" ; 
    if(!table) table = G4Material::GetMaterialTable();

    G4int n = G4Material::GetNumberOfMaterials();
    cout << " G4DAEMaterialMap numOfMaterials " << n << endl ;

    for(G4int i=0 ; i < n ; i++)
    {
         G4Material* m = (*table)[i];
         G4int index = m->GetIndex();
         G4String name = m->GetName();
        
         m_map[name] = index ; 

         cout << " material " 
              << " i " << i
              << " index " << index 
              << " name"   << name 
              << endl; 
    } 
}


string G4DAEMaterialMap::GetKey()
{
    return m_key ; 
}


void G4DAEMaterialMap::Print(const char* msg)
{
    cout  << msg 
          << " key " << m_key 
          << " min " << GetMinIndex() 
          << " max " << GetMaxIndex()
          << endl ; 

    for(int i=GetMinIndex() ; i <= GetMaxIndex() ; i++ )
    {
       cout << setw(40) << FindName(i) << " : " << i  << endl ;
    }
}

int G4DAEMaterialMap::GetMaxIndex()
{
    int imax = -1 ; 
    for(SI_t::iterator it=m_map.begin() ; it != m_map.end() ; it++ )
    {
        if(it->second > imax) imax = it->second ; 
    }    
    return imax ; 
}

int G4DAEMaterialMap::GetMinIndex()
{
    int imin = 1000000 ; 
    for(SI_t::iterator it=m_map.begin() ; it != m_map.end() ; it++ )
    {
        if(it->second < imin) imin = it->second ; 
    }    
    return imin ; 
}

int G4DAEMaterialMap::FindIndex(const char* name)
{
    string key(name);
    if( m_map.find(key) != m_map.end() )
    {
        return m_map[key];
    }
    else
    {
        return -1 ;
    }
}


std::string G4DAEMaterialMap::FindName(int index)
{
    for(SI_t::iterator it=m_map.begin() ; it != m_map.end() ; it++ )
    {
        if( it->second == index ) return it->first ;
    }
    string empty ; 
    return empty ; 
}



int* G4DAEMaterialMap::MakeLookupArray(G4DAEMaterialMap* a, G4DAEMaterialMap* b) // from a->b
{
     // a:geant4   
     // b:chroma      (a expected to contain all of b)
     //
     int mi = a->GetMinIndex();     
     int mx = a->GetMaxIndex();     
     assert(mi == 0);
     assert(mx > 0);

     int* a2b = new int[mx+1] ; 

     for(int ia=mi ; ia <= mx ; ia++ )
     {
         string an = a->FindName(ia);
         int ib = b->FindIndex(an.c_str());
         string bn ; 
         if( ib > -1 )
         {
             bn = b->FindName(ib);
             assert( an == bn);
         }

        /*
         cout 
              << " ia " << setw(3) << ia 
              << " an " << setw(40) << an 
              << " ib " << setw(3) << ib 
              << " bn " << setw(40) << bn
              << endl ; 
         */

         a2b[ia] = ib ; 
     }

     DumpLookupArray(a, b, a2b);
     return a2b ; 
}


void G4DAEMaterialMap::DumpLookupArray( G4DAEMaterialMap* a, G4DAEMaterialMap* b, int* a2b )
{
    cout << "G4DAEMaterialMap::DumpLookupArray " 
         << " a " << a->GetKey() 
         << " b " << b->GetKey() 
         << endl ; 

    for( int ia=0 ; ia <= a->GetMaxIndex() ; ia++ )
    {   
        int ib = a2b[ia];
        cout << " ia "  << setw(3) << ia 
             << " an "  << setw(40) << a->FindName(ia)
             << " ib "  << setw(3) << ib  
             << " bn "  << setw(40) << b->FindName(ib)
             << endl ;
    }   
}   



G4DAEMaterialMap::~G4DAEMaterialMap()
{
}

 
