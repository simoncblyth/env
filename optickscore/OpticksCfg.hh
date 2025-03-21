#pragma once
#include "Opticks.hh"
#include "BCfg.hh"

#include "OKCORE_API_EXPORT.hh"


template <class Listener>
class OKCORE_API OpticksCfg : public BCfg {
  public:
     OpticksCfg(const char* name, Listener* listener, bool live);
  public:
     void dump(const char* msg="OpticksCfg::dump");


     const std::string& getSize();
     const std::string& getPosition();
     const std::string& getLogName();
     const std::string& getConfigPath();
     const std::string& getEventTag();
     const std::string& getEventCat();
     const std::string& getLiveLine();

     const std::string& getExportConfig();
     const std::string& getTorchConfig();

     const std::string& getG4GunConfig();
     const std::string& getG4IniMac();
     const std::string& getG4RunMac();
     const std::string& getG4FinMac();

     const std::string& getTestConfig();
     const std::string& getStateTag();

     const std::string& getZExplodeConfig();
     const std::string& getMeshVersion();
     const std::string& getISlice();
     const std::string& getFSlice();
     const std::string& getPSlice();
     const std::string& getPrintIndex();
     const std::string& getBuilder();
     const std::string& getTraverser();

     float        getEpsilon(); 
     int          getRngMax(); 
     int          getBounceMax(); 
     int          getRecordMax(); 
     int          getTimeMax(); 
     int          getAnimTimeMax(); 
     int          getInterpolatedViewPeriod(); 
     int          getOrbitalViewPeriod(); 
     int          getTrackViewPeriod(); 
     int          getAnimatorPeriod(); 
     int          getRepeatIndex(); 
     int          getRestrictMesh(); 
     int          getAnalyticMesh(); 
     int          getModulo(); 
     int          getOverride(); 
     int          getDebugIdx(); 
     int          getStack(); 
     int          getNumPhotonsPerG4Event(); 
     int          getLoaderVerbosity(); 
     int          getMeshVerbosity(); 
private:
     void init();
private:
     Listener*   m_listener ; 
     std::string m_size ;
     std::string m_position ;
     std::string m_logname ;
     std::string m_event_cat ;
     std::string m_event_tag ;
     std::string m_liveline ;

     std::string m_configpath ;
     std::string m_exportconfig ;
     std::string m_torchconfig ;

     std::string m_g4gunconfig ;
     std::string m_g4inimac ;
     std::string m_g4runmac ;
     std::string m_g4finmac ;

     std::string m_testconfig ;
     std::string m_state_tag ;

     std::string m_zexplodeconfig ;
     std::string m_meshversion ;
     std::string m_islice ;
     std::string m_fslice ;
     std::string m_pslice ;
     std::string m_pindex ;
     std::string m_builder ;
     std::string m_traverser  ;

     float       m_epsilon ; 
     int         m_rngmax ; 
     int         m_bouncemax ; 
     int         m_recordmax ; 
     int         m_timemax ; 
     int         m_animtimemax ; 
     int         m_animator_period ; 
     int         m_ivperiod ; 
     int         m_ovperiod ; 
     int         m_tvperiod ; 
     int         m_repeatidx ; 
     int         m_restrictmesh; 
     int         m_analyticmesh; 
     int         m_modulo ; 
     int         m_override ; 
     int         m_debugidx ; 
     int         m_stack ; 
     int         m_num_photons_per_g4event;
     int         m_loaderverbosity ; 
     int         m_meshverbosity ; 
};


