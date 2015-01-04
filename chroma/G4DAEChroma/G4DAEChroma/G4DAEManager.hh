#ifndef G4DAEMANAGER_H
#define G4DAEMANAGER_H 1


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
    size_t GetFlags();
    bool HasFlag(size_t flg);
    bool HasFlag(const char* name);
    std::string Flags();

public:
    size_t GetStatus(size_t flg);
    size_t GetStatus(const char* name);

public:
    void Start(size_t flg);
    void Stop(size_t flg);

public:
    G4DAEMetadata* GetConfig();
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

    // task names corresponding to the flags 
    const char* m_name[MAXTASK] ;

    // task status, eg disabled, active, ...
    size_t m_status[MAXTASK];
 

};


#endif
