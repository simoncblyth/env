#include "G4DAEChroma/G4DAEManager.hh"

#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEMap.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std ; 

const size_t  G4DAEManager::MAXTASK ; 

G4DAEManager::G4DAEManager(const char* configkey)  : m_flags(0), m_config(NULL)
{
   Initialize(configkey);
}

G4DAEManager::~G4DAEManager()
{
}

void G4DAEManager::Initialize(const char* configkey)
{
    ZeroConfig();
    ZeroResults();
    LoadConfig(configkey);
    LoadFlags("/FLAGS");
    DumpConfig();
}

G4DAEMetadata* G4DAEManager::GetConfig()
{
    return m_config ;
}

void G4DAEManager::LoadConfig(const char* configkey)
{
    const char* jspath = getenv(configkey);
    if(!jspath)
    {
        printf("G4DAEManager::LoadConfig missing envvar %s \n", configkey );
        return ; 
    } 

    printf("G4DAEManager::LoadConfig configkey %s jspath %s \n", configkey, jspath);
    G4DAEMetadata* config = G4DAEMetadata::CreateFromFile(jspath);
    if(!config)
    {
        printf("G4DAEManager::LoadConfig missing config file  %s \n", jspath );
        return ; 
    } 
    m_config = config ;
}


void G4DAEManager::LoadFlags(const char* cfgpath)
{
    if(!m_config) return ;

    Map_t flags = m_config->GetRawMap(cfgpath);

    for(Map_t::iterator it=flags.begin() ; it != flags.end() ; it++ )
    {   
        string key = it->first ;
        string val = it->second ;
        int ival = atoi(val.c_str());
        assert(ival >= 0 && ival < MAXTASK);
        m_name[ival] = strdup(key.c_str()); 
        //printf(" %20s : %s : %d \n", key.c_str(), val.c_str(), ival );
    }   
}

void G4DAEManager::DumpConfig(const char* msg)
{
    cout << msg << endl ; 
    if(!m_config)
    {
        cout << "CONFIG HAS NOT BEEN LOADED" << endl ; 
        return ; 
    }
    for(size_t i=0 ; i<MAXTASK ; i++)
    {
        printf(" %zu : %s \n", i, m_name[i] );
    }
}


void G4DAEManager::ZeroConfig()
{
    for(size_t i=0 ; i<MAXTASK ; i++)
    {
        if(m_name[i]) free((void*)m_name[i]);
        m_name[i] = NULL ; 
    }
}

void G4DAEManager::ZeroResults()
{
    for(size_t i=0 ; i<MAXTASK ; i++)
    {
        m_count[i] = 0 ;
        m_start[i] = 0. ;
        m_stop[i] = 0. ;
        m_duration[i] = 0. ;
    }
}

string G4DAEManager::Flags()
{
    // presentation of flag settings
    vector<string> elem ; 
    for(size_t i=0 ; i<MAXTASK ; i++)
    {
        const char* name = m_name[i] ;
        if(HasFlag(i) && name ) elem.push_back(string(name)) ;
    }
    return join(elem, '\n') ; 
}

size_t G4DAEManager::FindFlag(const char* flag )
{
    size_t iflag = 0 ; 
    for(size_t i=0 ; i<MAXTASK ; i++)
    {
        if(m_name[i] && strcmp(flag, m_name[i]) == 0)  iflag = i ;
    }
    return iflag ;
}

size_t G4DAEManager::ParseFlags(string sflags, char delim)
{
    //cout << "G4DAEManager::ParseFlags " << sflags << endl ; 
    typedef vector<string> Vec_t ;
    Vec_t elems ; 
    split(elems, sflags.c_str(), delim);

    size_t flags = 0 ; 
    for(Vec_t::iterator it=elems.begin() ; it!=elems.end() ; it++)
    {
        string elem = *it;
        size_t ibit = FindFlag( elem.c_str() );
        if(ibit == 0)
        {
            cout << "G4DAEManager::ParseFlags : IGNORING UNKNOWN FLAG  "
                 << " elem " << elem 
                 << " ibit " << ibit 
                 << endl ; 
            continue ; 
        }
        flags |= ( 1 << ibit ) ; 
    }
    return flags ; 
}


// setters

void G4DAEManager::SetFlags(string flags)
{
     size_t _flags = ParseFlags(flags);
     SetFlags(_flags);
}
void G4DAEManager::SetFlags(size_t flags)
{
    m_flags = flags ; 
}
void G4DAEManager::AddFlags(size_t flags)
{
    m_flags |= flags ; 
}
void G4DAEManager::AddFlags(string flags)
{
    size_t _flags = ParseFlags(flags);
    AddFlags(_flags); 
}


// getters

size_t G4DAEManager::GetFlags()
{
    return m_flags ; 
}
bool G4DAEManager::HasFlag(size_t flg)
{
    // use this for inner loops
    return m_flags & (1 << flg) ; 
}
bool G4DAEManager::HasFlag(const char* name)
{
    size_t flg = FindFlag(name);
    return flg && HasFlag(flg);
}





