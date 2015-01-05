#ifndef G4DAEMANAGER_H
#define G4DAEMANAGER_H 1

#include "G4DAEChroma/G4DAEMap.hh" 
#include <string>

class G4DAEMetadata ; 

class G4DAEManager
{
public:
    static const size_t MAXTASK = 32 ; 
protected:
    G4DAEManager(const char* configkey);
    void Initialize(const char* configkey);
public:
    virtual ~G4DAEManager();

public:
    // a task is a flag with status > 0  
    size_t FindTask(size_t flg); 
    size_t FindTask(const char* name); 

public:
    size_t ParseFlags(std::string sflags, char delim=',');
    size_t FindFlag(const char* name);
    void SetFlags(size_t flags);
    void SetFlags(std::string flags);
    void AddFlags(size_t flags);
    void AddFlags(std::string flags);

public:
    // TODO:elimnate status/flags duplication
    void AddStatus(size_t flags);
    void SetStatus(size_t flags);

public:
    const char* GetName(size_t flg);
    size_t GetFlags();
    bool HasFlag(size_t flg);
    bool HasFlag(const char* name);
    std::string Flags();

public:
    size_t GetStatus(size_t flg);
    size_t GetStatus(const char* name);

public:
    void Start(size_t flg, size_t verbosity=0);
    void Stop(size_t flg, size_t verbosity=0);
    void Skip(size_t flg, size_t verbosity=0);
    void Register(size_t flg, size_t modulo=0);

    void Start(const char* name, size_t verbosity=0);
    void Stop(const char* name, size_t verbosity=0);
    void Skip(const char* name, size_t verbosity=0);

    void Stamp(const char* name, size_t verbosity=0);


public:
    G4DAEMetadata* GetConfig();
    G4DAEMetadata* GetResults();
    void UpdateResults(); // from maps and arrays into the metadata

    void DumpConfig(const char* msg="G4DAEManager::DumpConfig");
    void DumpResults(const char* msg="G4DAEManager::DumpResults");
    void LoadConfig(const char* configkey);
    void LoadMap(const char* cfgpath, char dest);
    void ZeroConfig();
    void ZeroResults();

private:

    // control bitfield
    size_t m_flags ;  

    // flags names and numbers
    G4DAEMetadata* m_config ; 

    // timestamps and durations
    G4DAEMetadata* m_results ; 

    // last realtime start
    double m_start[MAXTASK] ; 

    // last realtime stop
    double m_stop[MAXTASK] ; 

    // acculated realtime durations stop - start 
    double m_duration[MAXTASK] ; 

    // per count average of acculated realtime durations  
    double m_average[MAXTASK] ; 

    // start counts
    size_t m_count[MAXTASK];

    // counters without timing 
    size_t m_register[MAXTASK];

    // task names corresponding to the flags 
    const char* m_name[MAXTASK] ;

    // task status, eg disabled, active, ...
    size_t m_status[MAXTASK];
 
    // timestamps
    Map_t m_timestamp ; 



};


#endif
