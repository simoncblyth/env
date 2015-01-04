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
    size_t ParseFlags(std::string sflags, char delim=',');
    size_t FindFlag(const char* name);
    void SetFlags(size_t flags);
    void SetFlags(std::string flags);
    void AddFlags(size_t flags);
    void AddFlags(std::string flags);

public:
    size_t GetFlags();
    bool HasFlag(size_t flags);
    bool HasFlag(const char* name);
    std::string Flags();


public:
    G4DAEMetadata* GetConfig();
    void DumpConfig(const char* msg="G4DAEManager::DumpConfig");
    void LoadConfig(const char* configkey);
    void LoadFlags(const char* cfgpath);
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

    // start counts
    size_t m_count[MAXTASK];

    // task names corresponding to the flags 
    const char* m_name[MAXTASK] ;

 

};


#endif
