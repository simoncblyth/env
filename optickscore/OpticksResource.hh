#pragma once

#include <map>
#include <string>

class Opticks ; 
class OpticksQuery ; 
class OpticksColors ; 
class OpticksFlags ; 
class OpticksAttrSeq ;

class NEnv ; 

class Types ;
class Typ ;

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksResource {
    private:
       static const char* JUNO ; 
       static const char* DAYABAY ; 
       static const char* DPIB ; 
       static const char* OTHER ; 
    private:
       static const char* PREFERENCE_BASE  ;
    public:
       static const char* DEFAULT_GEOKEY ;
       static const char* DEFAULT_QUERY ;
       static const char* DEFAULT_CTRL ;
       static const char* DEFAULT_MESHFIX ;
       static const char* DEFAULT_MESHFIX_CFG ;
    public:
       static bool existsFile(const char* path);
       static bool existsFile(const char* dir, const char* name);
       static bool existsDir(const char* path);
    public:
       OpticksResource(Opticks* opticks=NULL, const char* envprefix="OPTICKS_", const char* lastarg=NULL);
       bool isValid();
    private:
       void init();
       void adoptInstallPrefix();
       void readG4Environment();
       void readEnvironment();
       void readMetadata();
       void identifyGeometry();
       void assignDetectorName();
       void setValid(bool valid);
    public:
       const char* getInstallPrefix();
       const char* getIdPath();
       const char* getIdFold();  // parent directory of idpath containing g4_00.dae
       const char* getIdBase();  // parent directory of idfold, typically the "export" folder
       const char* getDetectorBase();  // eg /usr/local/opticks/opticksdata/export/DayaBay 
    public:
       std::string getRelativePath(const char* path); 
       std::string getRelativePath(const char* name, unsigned int ridx);
       std::string getObjectPath(const char* name, unsigned int ridx);
       std::string getDetectorPath(const char* name, unsigned int ridx);
       std::string getMergedMeshPath(unsigned int ridx);
       std::string getPmtPath(unsigned int index, bool relative=false);
       std::string getPropertyLibDir(const char* name);
    public:
       std::string getPreferenceDir(const char* type, const char* udet=NULL, const char* subtype=NULL);
       bool loadPreference(std::map<std::string, std::string>& mss, const char* type, const char* name);
       bool loadPreference(std::map<std::string, unsigned int>& msu, const char* type, const char* name);
    public:
       bool loadMetadata(std::map<std::string, std::string>& mdd, const char* path);
       void dumpMetadata(std::map<std::string, std::string>& mdd);
       bool hasMetaKey(const char* key);
       const char* getMetaValue(const char* key);
    public:
       const char* getEnvPrefix();
       bool idPathContains(const char* s); 
       void Summary(const char* msg="OpticksResource::Summary");
       void Dump(const char* msg="OpticksResource::Dump");
    public:
       const char* getDAEPath();
       const char* getGDMLPath();
       const char* getQueryString();
       const char* getCtrl();
    public:
       OpticksQuery* getQuery();
       OpticksColors* getColors();
       OpticksFlags*  getFlags();
       OpticksAttrSeq* getFlagNames();
       std::map<unsigned int, std::string> getFlagNamesMap();
   public:
       Types*         getTypes();
       Typ*           getTyp();
    private:
       std::string makeSidecarPath(const char* path, const char* styp=".dae", const char* dtyp=".ini");
    public:
       const char* getMetaPath();
    public:
       const char* getMeshfix();
       const char* getMeshfixCfg();
       glm::vec4   getMeshfixFacePairingCriteria();
    public:
       const char* getDetector();
       const char* getDetectorName();
       bool        isJuno();
       bool        isDayabay();
       bool        isPmtInBox();
       bool        isOther();
   private:
       Opticks*    m_opticks ; 
       const char* m_envprefix ; 
       const char* m_lastarg ; 
       const char* m_install_prefix ;  // from OpticksCMakeConfig header
   private:
       // results of readEnvironment
       const char* m_geokey ;
       const char* m_daepath ;
       const char* m_gdmlpath ;
       const char* m_query_string ;
       const char* m_ctrl ;
       const char* m_metapath ;
       const char* m_meshfix ;
       const char* m_meshfixcfg ;
       const char* m_idpath ;
       const char* m_idfold ;
       const char* m_idbase ;
       const char* m_digest ;
       bool        m_valid ; 
   private:
       OpticksQuery*  m_query ;
       OpticksColors* m_colors ;
       OpticksFlags*  m_flags ;
       OpticksAttrSeq* m_flagnames ;
       Types*         m_types ;
       Typ*           m_typ ;
       NEnv*          m_g4env ; 
   private:
       // results of identifyGeometry
       bool        m_dayabay ; 
       bool        m_juno ; 
       bool        m_dpib ; 
       bool        m_other ; 
       const char* m_detector ;
       const char* m_detector_name ;
       const char* m_detector_base ;
       
   private:
      std::map<std::string, std::string> m_metadata ;  
};


#include "OKCORE_TAIL.hh"

