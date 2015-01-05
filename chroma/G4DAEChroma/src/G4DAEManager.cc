#include "G4DAEChroma/G4DAEManager.hh"

#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEMap.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std ; 

const size_t  G4DAEManager::MAXTASK ; 

G4DAEManager::G4DAEManager(const char* configkey)  : m_flags(0), m_config(NULL), m_results(NULL)
{
   Initialize(configkey);
}

G4DAEManager::~G4DAEManager()
{
}

void G4DAEManager::ZeroConfig()
{
    for(size_t i=0 ; i<MAXTASK ; i++)
    {
        m_name[i] = NULL ; 
        m_status[i] = 0 ;   // hmm parallel to m_flags bitfield
    }
}

void G4DAEManager::ZeroResults()
{
    for(size_t i=0 ; i<MAXTASK ; i++)
    {
        m_count[i] = 0 ;
        m_register[i] = 0 ;
        m_start[i] = 0. ;
        m_stop[i] = 0. ;
        m_duration[i] = 0. ;
        m_average[i] = 0. ;
    }
}


void G4DAEManager::Initialize(const char* configkey)
{
    ZeroConfig();
    ZeroResults();
    LoadConfig(configkey);
    LoadMap("/FLAGS", 'f');
    LoadMap("/STATUS", 's');
    DumpConfig();
}

G4DAEMetadata* G4DAEManager::GetConfig()
{
    return m_config ;
}
G4DAEMetadata* G4DAEManager::GetResults()
{
    if(!m_results) m_results = new G4DAEMetadata("{}");
    return m_results ;
}

void G4DAEManager::UpdateResults()
{
    G4DAEMetadata* results = GetResults();
    results->AddMap("timestamp",m_timestamp);
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


void G4DAEManager::LoadMap(const char* cfgpath, char dest)
{
    if(!m_config) return ;

    Map_t map = m_config->GetRawMap(cfgpath);

    int imax = MAXTASK ;
    for(Map_t::iterator it=map.begin() ; it != map.end() ; it++ )
    {   
        string key = it->first ;
        string val = it->second ;
        int ival = atoi(val.c_str());
        assert(ival >= 0 && ival < imax);

        if(dest == 'f')
        {
            // establish the names of the flags 
            m_name[ival]   = strdup(key.c_str()); 
        } 
        else if (dest == 's')
        {
            // set status of the named flag 
            size_t iflag = FindFlag(key.c_str());
            if(iflag == 0)
            {
                printf("G4DAEManager::LoadMap no such key %s \n", key.c_str());
                continue ; 
            }
            m_status[iflag] = ival ; 
            if(ival > 0) AddFlags( 1 << iflag ) ;    // TODO: get rid of duplication between m_status and m_flags
        }
                      
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
        printf(" %2zu : %zu : %s \n", i, m_status[i], m_name[i] );
    }
}

void G4DAEManager::DumpResults(const char* msg)
{
    cout << msg << endl ; 
    if(!m_config)
    {
        cout << "CONFIG HAS NOT BEEN LOADED" << endl ; 
        return ; 
    }
    for(size_t i=0 ; i<MAXTASK ; i++)
    {
        printf(" %2zu : %zu : %50s : %7zu %7zu : %10.4f  %10.4f  \n", 
              i, 
              m_status[i], 
              m_name[i], 
              m_register[i], 
              m_count[i], 
              m_duration[i], 
              m_average[i] );
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


const char* G4DAEManager::GetName(size_t flg)
{
    return ( flg < MAXTASK ) ? m_name[flg] : NULL ;
}



void G4DAEManager::Skip(const char* name, size_t verbosity)
{
    size_t task = FindFlag(name);
    Skip(task, verbosity);
}
void G4DAEManager::Start(const char* name, size_t verbosity)
{
    size_t task = FindFlag(name);
    Start(task, verbosity);
}

void G4DAEManager::Stop(const char* name, size_t verbosity)
{
    size_t task = FindFlag(name);
    Stop(task, verbosity);
}

void G4DAEManager::Stamp(const char* name, size_t verbosity)
{
    string stamp =  G4DAEMetadata::TimeStampLocal();
    if(verbosity > 0)
    {
        printf("G4DAEManager::Stamp %s %s \n", name, stamp.c_str());
    }
    m_timestamp[name] = stamp ;  
}

void G4DAEManager::Skip(size_t task, size_t verbosity)
{
    if(verbosity>0)
    {
        const char* name = GetName(task);
        printf("G4DAEManager::Skip %zu %s \n", task, name);
    }
}

void G4DAEManager::Register(size_t task, size_t modulo)
{
    if(task >= MAXTASK) return ; 

    if(modulo > 0 && m_register[task] % modulo == 0)
    {
        const char* name = GetName(task);
        printf("G4DAEManager::Register task %zu modulo %zu register %zu name %s \n", task, modulo, m_register[task], name);
    }

    m_register[task] += 1 ; 

    

}


void G4DAEManager::Start(size_t task, size_t verbosity)
{
    if(verbosity>0)
    {
        const char* name = GetName(task);
        printf("G4DAEManager::Start %zu %s \n", task, name);
    }

    if(task == 0 || task >= MAXTASK)
    {
        printf("G4DAEManager::Start invalid task %zu \n", task);
        return ;
    }

    m_start[task] = G4DAEMetadata::RealTime();
}
void G4DAEManager::Stop(size_t task, size_t verbosity)
{
    if(task == 0 || task >= MAXTASK)
    {
        printf("G4DAEManager::Stop invalid task %zu \n", task);
        return ;
    }

    m_count[task] += 1 ; 
    m_stop[task] = G4DAEMetadata::RealTime();
    m_duration[task] += m_stop[task] - m_start[task] ; 
    m_average[task] = m_duration[task]/m_count[task] ; 

    if(verbosity>0)
    {
        const char* name = GetName(task);
        printf("G4DAEManager::Stop %zu %s \n", task, name);
    }

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
    SetStatus(flags);
}

void G4DAEManager::SetStatus(size_t flgs)
{
    for(size_t i=0 ; i < MAXTASK ; i++)
    {
        m_status[i] = (flgs & ( 1 << i )) ? 1 : 0 ;
    }
}
void G4DAEManager::AddStatus(size_t flgs)
{
    for(size_t i=0 ; i < MAXTASK ; i++)
    {
        if(flgs & ( 1 << i )) m_status[i] = 1 ;  
    }
}

void G4DAEManager::AddFlags(size_t flags)
{
    m_flags |= flags ; 
    AddStatus(flags);
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
    return m_flags & (1 << flg) ; 
}
bool G4DAEManager::HasFlag(const char* name)
{
    size_t flg = FindFlag(name);
    return flg && HasFlag(flg);
}


size_t G4DAEManager::FindTask(const char* name)
{
    size_t flg = FindFlag(name);
    return FindTask(flg);
}

size_t G4DAEManager::FindTask(size_t flg)
{
    // a task is a flag with status > 0 
    return ( HasFlag(flg) && GetStatus(flg) > 0) ? flg : 0 ; 
}


size_t G4DAEManager::GetStatus(const char* name)
{
    size_t flg = FindFlag(name);
    return GetStatus(flg); 
}
size_t G4DAEManager::GetStatus(size_t flg)
{
    return m_status[flg];
}




