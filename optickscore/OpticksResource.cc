#define LL info

#include <cstring>
#include <cassert>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


// sysrap
#include "SDigest.hh"
#include "SSys.hh"
// brap-
#include "BFile.hh"
#include "BStr.hh"
#include "PLOG.hh"

// npy-
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "Map.hpp"
#include "NEnv.hpp"
#include "Typ.hpp"
#include "Types.hpp"
#include "Index.hpp"


// CMake generated defines from binary_dir/inc
#include "OpticksCMakeConfig.hh"   

#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksQuery.hh"
#include "OpticksColors.hh"
#include "OpticksFlags.hh"
#include "OpticksAttrSeq.hh"


const char* OpticksResource::JUNO    = "juno" ; 
const char* OpticksResource::DAYABAY = "dayabay" ; 
const char* OpticksResource::DPIB    = "PmtInBox" ; 
const char* OpticksResource::OTHER   = "other" ; 

const char* OpticksResource::PREFERENCE_BASE = "$HOME/.opticks" ; 

const char* OpticksResource::DEFAULT_GEOKEY = "DAE_NAME_DYB" ; 
const char* OpticksResource::DEFAULT_QUERY = "range:3153:12221" ; 
const char* OpticksResource::DEFAULT_CTRL = "volnames" ; 
const char* OpticksResource::DEFAULT_MESHFIX = "iav,oav" ; 
const char* OpticksResource::DEFAULT_MESHFIX_CFG = "100,100,10,-0.999" ; 


OpticksResource::OpticksResource(Opticks* opticks, const char* envprefix, const char* lastarg) 
    :
       m_opticks(opticks),
       m_envprefix(strdup(envprefix)),
       m_lastarg(lastarg ? strdup(lastarg) : NULL),
       m_install_prefix(NULL),

       m_geokey(NULL),
       m_daepath(NULL),
       m_gdmlpath(NULL),
       m_query_string(NULL),
       m_ctrl(NULL),
       m_metapath(NULL),
       m_meshfix(NULL),
       m_meshfixcfg(NULL),
       m_idpath(NULL),
       m_idfold(NULL),
       m_idbase(NULL),
       m_digest(NULL),
       m_valid(true),
       m_query(NULL),
       m_colors(NULL),
       m_flags(NULL),
       m_flagnames(NULL),
       m_types(NULL),
       m_typ(NULL),
       m_g4env(NULL),
       m_dayabay(false),
       m_juno(false),
       m_dpib(false),
       m_other(false),
       m_detector(NULL),
       m_detector_name(NULL)
{
    init();
}



const char* OpticksResource::getInstallPrefix()
{
    return m_install_prefix ; 
}



void OpticksResource::setValid(bool valid)
{
    m_valid = valid ; 
}
bool OpticksResource::isValid()
{
   return m_valid ; 
}
const char* OpticksResource::getIdPath()
{
    return m_idpath ;
}
const char* OpticksResource::getIdFold()
{
    return m_idfold ;
}
const char* OpticksResource::getIdBase()
{
    return m_idbase ;
}
const char* OpticksResource::getDetectorBase()
{
    return m_detector_base ;
}




const char* OpticksResource::getEnvPrefix()
{
    return m_envprefix ;
}
const char* OpticksResource::getDAEPath()
{
    return m_daepath ;
}
const char* OpticksResource::getGDMLPath()
{
    return m_gdmlpath ;
}
const char* OpticksResource::getMetaPath()
{
    return m_metapath ;
}


const char* OpticksResource::getQueryString()
{
    return m_query_string ;
}
OpticksQuery* OpticksResource::getQuery()
{
    return m_query ;
}


const char* OpticksResource::getCtrl()
{
    return m_ctrl ;
}
const char* OpticksResource::getMeshfix()
{
    return m_meshfix ;
}
const char* OpticksResource::getMeshfixCfg()
{
    return m_meshfixcfg ;
}


const char* OpticksResource::getDetector()
{
    return m_detector ;
}
const char* OpticksResource::getDetectorName()
{
    return m_detector_name ;
}




bool OpticksResource::isJuno()
{
   return m_juno ; 
}
bool OpticksResource::isDayabay()
{
   return m_dayabay ; 
}
bool OpticksResource::isPmtInBox()
{
   return m_dpib ; 
}
bool OpticksResource::isOther()
{
   return m_other ; 
}



bool OpticksResource::idPathContains(const char* s)
{
    bool ret = false ; 
    if(m_idpath)
    {
        std::string idp(m_idpath);
        std::string ss(s);
        ret = idp.find(ss) != std::string::npos ;
    }
    return ret ; 
}

std::string OpticksResource::getRelativePath(const char* path)
{
    if(strncmp(m_idpath, path, strlen(m_idpath)) == 0)
    {
        return path + strlen(m_idpath) + 1 ; 
    }
    else
    {
        return path ;  
    }
}






void OpticksResource::init()
{
   LOG(trace) << "OpticksResource::init" ; 

   adoptInstallPrefix();
   readG4Environment();
   readEnvironment();
   readMetadata();
   identifyGeometry();
   assignDetectorName(); 

   LOG(trace) << "OpticksResource::init DONE" ; 
}


void OpticksResource::adoptInstallPrefix()
{
   m_install_prefix = strdup(OPTICKS_INSTALL_PREFIX) ; 
   
   const char* key = "INSTALL_PREFIX" ; 

   int rc = SSys::setenvvar(m_envprefix, key, m_install_prefix, true ); 

   LOG(trace) << "OpticksResource::adoptInstallPrefix " 
              << " install_prefix " << m_install_prefix  
              << " envprefix " << m_envprefix  
              << " key " << key
              << " rc " << rc
              ;  
 
   assert(rc==0); 

   // The CMAKE_INSTALL_PREFIX from opticks-;opticks-cmake 
   // is set to the result of the opticks-prefix bash function 
   // at configure time.
   // This is recorded into a config file by okc-/CMakeLists.txt 
   // and gets compiled into the OpticksCore library.
   //
   // Canonically it is :  /usr/local/opticks 
}


void OpticksResource::identifyGeometry()
{
   // TODO: somehow extract detector name from the exported file metadata or sidecar


   m_juno     = idPathContains("export/juno") ;
   m_dayabay  = idPathContains("export/DayaBay") ;
   m_dpib     = idPathContains("export/dpib") ;

   if(m_juno == false && m_dayabay == false && m_dpib == false )
   {
       const char* detector = getMetaValue("detector") ;
       if(detector)
       {

           if(     strcmp(detector, DAYABAY) == 0) m_dayabay = true ; 
           else if(strcmp(detector, JUNO)    == 0) m_juno = true ; 
           else if(strcmp(detector, DPIB)    == 0) m_dpib = true ; 
           else 
                 m_other = true ;

           LOG(trace) << "OpticksResource::identifyGeometry" 
                      << " metavalue detector " <<  detector 
                      ; 
       }
       else
           m_other = true ;
   }


   assert( m_juno ^ m_dayabay ^ m_dpib ^ m_other ); // exclusive-or
   
   if(m_juno)    m_detector = JUNO ; 
   if(m_dayabay) m_detector = DAYABAY ; 
   if(m_dpib)    m_detector = DPIB ; 
   if(m_other)   m_detector = OTHER ; 

}

void OpticksResource::assignDetectorName()
{
   std::map<std::string, std::string> detname ; 
   detname["juno"]    = "juno" ;
   detname["dayabay"] = "DayaBay" ;
   detname["dpib"]    = "dpib" ;

   if(m_detector && detname.count(m_detector) == 1) m_detector_name =  strdup(detname[m_detector].c_str()) ; 

   if(m_detector_name && m_idbase )
   {
        std::string detbase = BFile::FormPath(m_idbase, m_detector_name);
        m_detector_base = strdup(detbase.c_str());
   }
}

void OpticksResource::readG4Environment()
{
    // NB this path needs to match that in g4-;g4-export-ini
    std::string g4ini = BFile::FormPath(m_install_prefix, "externals/config/geant4.ini" ) ;

    if(BFile::ExistsFile(g4ini.c_str()))
    {
        LOG(info) << "OpticksResource::readG4Environment" 
                  << " from " << g4ini
                  ;

         m_g4env = NEnv::load(g4ini.c_str()); 
         m_g4env->setEnvironment();
    }
    else
    {
        LOG(warning) << "OpticksResource::readG4Environment"
                     << " MISSING FILE " << g4ini 
                     << " (create it with bash functions: g4-;g4-export-ini ) " 
                     ;

    }
    NEnv::dumpEnvironment();
}


void OpticksResource::readEnvironment()
{
/*

:param envprefix: of the required envvars, eg with "GGEOVIEW_" need:

*path* 
     identifies the source geometry G4DAE exported file

*query*
     string used to select volumes from the geometry 

*idpath* 
     directory name based on *path* incorporating a hexdigest of the *query* 
     this directory is used to house:

     * geocache of NPY persisted buffers of the geometry
     * json metadata files
     * bookmarks

*ctrl*
     not currently used?

*/

    m_geokey = SSys::getenvvar(m_envprefix, "GEOKEY", DEFAULT_GEOKEY);
    m_daepath = getenv(m_geokey);

    if(m_daepath == NULL)
    {
        if(m_lastarg && existsFile(m_lastarg))
        {
            m_daepath = m_lastarg ; 
            LOG(warning) << "OpticksResource::readEnvironment"
                         << " MISSING ENVVAR "
                         << " geokey " << m_geokey 
                         << " lastarg " << m_lastarg
                         << " daepath " << m_daepath
                         ;
        }
    }

    if(m_daepath == NULL)
    {
        LOG(warning) << "OpticksResource::readEnvironment"
                     << " NO DAEPATH "
                     << " geokey " << m_geokey 
                     << " lastarg " << ( m_lastarg ? m_lastarg : "NULL" )
                     << " daepath " << ( m_daepath ? m_daepath : "NULL" )
                     ;
 
        //assert(0);
        setValid(false);
    } 
    else
    {
         std::string metapath = makeSidecarPath(m_daepath, ".dae", ".ini");
         m_metapath = strdup(metapath.c_str());
         std::string gdmlpath = makeSidecarPath(m_daepath, ".dae", ".gdml");
         m_gdmlpath = strdup(gdmlpath.c_str());
    }


    m_query_string = SSys::getenvvar(m_envprefix, "QUERY", DEFAULT_QUERY);
    m_ctrl         = SSys::getenvvar(m_envprefix, "CTRL", DEFAULT_CTRL);
    m_meshfix      = SSys::getenvvar(m_envprefix, "MESHFIX", DEFAULT_MESHFIX);
    m_meshfixcfg   = SSys::getenvvar(m_envprefix, "MESHFIX_CFG", DEFAULT_MESHFIX_CFG);

    m_query = new OpticksQuery(m_query_string);
    std::string query_digest = SDigest::md5digest( m_query_string, strlen(m_query_string));

    m_digest = strdup(query_digest.c_str());
 
    // idpath incorporates digest of geometry selection envvar 
    // allowing to benefit from caching as vary geometry selection 
    // while still only having a single source geometry file.

    if(m_daepath)
    {
        std::string kfn = BStr::insertField( m_daepath, '.', -1 , m_digest );

        m_idpath = strdup(kfn.c_str());

        assert(SSys::setenvvar("","IDPATH", m_idpath, true )==0);  // uses putenv for windows mingw compat 

        // Where is IDPATH used ? 
        //    Mainly by NPY tests as a resource access workaround as NPY 
        //    is lower level than optickscore- so lacks its resource access machinery.
        //

    }
    else
    {
         // IDPATH envvar is last resort, but handy for testing
         m_idpath = getenv("IDPATH");
    }

    if(m_idpath)
    {
        std::string fold = BFile::ParentDir(m_idpath);
        m_idfold = strdup(fold.c_str());

        // Parent of the IDPATH is the IDFOLD, typically contains
        //
        //       g4_00.dae 
        //       g4_00.gdml 
        //       ChromaMaterialMap.json  


        std::string base = BFile::ParentDir(m_idfold);
        m_idbase = strdup(base.c_str());

    } 

    // DO NOT PRINT ANYTHING FROM HERE TO AVOID IDP CAPTURE PROBLEMS
}


void OpticksResource::readMetadata()
{
    if(m_metapath)
    {
         loadMetadata(m_metadata, m_metapath);
         //dumpMetadata(m_metadata);
    }
}

void OpticksResource::Dump(const char* msg)
{
    Summary(msg);

    std::string mmsp = getMergedMeshPath(0);
    std::string pmtp = getPmtPath(0);

    std::cerr << "mmsp(0) :" << mmsp << std::endl ;  
    std::cerr << "pmtp(0) :" << pmtp << std::endl ;  
}

void OpticksResource::Summary(const char* msg)
{
    std::cerr << msg << std::endl ; 
    const char* prefix = m_install_prefix ; 

    std::cerr << "prefix   : " <<  (prefix ? prefix : "NULL" ) << std::endl ; 
    std::cerr << "valid    : " <<  (m_valid ? "valid" : "NOT VALID" ) << std::endl ; 
    std::cerr << "envprefix: " <<  (m_envprefix?m_envprefix:"NULL") << std::endl; 
    std::cerr << "geokey   : " <<  (m_geokey?m_geokey:"NULL") << std::endl; 
    std::cerr << "daepath  : " <<  (m_daepath?m_daepath:"NULL") << std::endl; 
    std::cerr << "gdmlpath : " <<  (m_gdmlpath?m_gdmlpath:"NULL") << std::endl; 
    std::cerr << "metapath : " <<  (m_metapath?m_metapath:"NULL") << std::endl; 
    std::cerr << "query    : " <<  (m_query_string?m_query_string:"NULL") << std::endl; 
    std::cerr << "ctrl     : " <<  (m_ctrl?m_ctrl:"NULL") << std::endl; 
    std::cerr << "digest   : " <<  (m_digest?m_digest:"NULL") << std::endl; 
    std::cerr << "idpath   : " <<  (m_idpath?m_idpath:"NULL") << std::endl; 
    std::cerr << "idfold   : " <<  (m_idfold?m_idfold:"NULL") << std::endl; 
    std::cerr << "idbase   : " <<  (m_idbase?m_idbase:"NULL") << std::endl; 
    std::cerr << "detector : " <<  (m_detector?m_detector:"NULL") << std::endl; 
    std::cerr << "detector_name : " <<  (m_detector_name?m_detector_name:"NULL") << std::endl; 
    std::cerr << "detector_base : " <<  (m_detector_base?m_detector_base:"NULL") << std::endl; 
    std::cerr << "getPmtPath(0) : " <<  (m_detector_base?getPmtPath(0):"-") << std::endl; 
    std::cerr << "meshfix  : " <<  (m_meshfix ? m_meshfix : "NULL" ) << std::endl; 
    std::cerr << "------ from " << ( m_metapath ? m_metapath : "NULL" ) << " -------- " << std::endl ;  

    typedef std::map<std::string, std::string> SS ;
    for(SS::const_iterator it=m_metadata.begin() ; it != m_metadata.end() ; it++)
        std::cerr <<  std::setw(10) << it->first.c_str() << ":" <<  it->second.c_str() << std::endl  ;

}



glm::vec4 OpticksResource::getMeshfixFacePairingCriteria()
{
   //
   // 4 comma delimited floats specifying criteria for faces to be deleted from the mesh
   //
   //   xyz : face barycenter alignment 
   //     w : dot face normal cuts 
   //

    assert(m_meshfixcfg) ; 
    std::string meshfixcfg = m_meshfixcfg ;
    return gvec4(meshfixcfg);
}

std::string OpticksResource::getMergedMeshPath(unsigned int index)
{
    return getObjectPath("GMergedMesh", index);
}

std::string OpticksResource::getPmtPath(unsigned int index, bool relative)
{
    return relative ?
                        getRelativePath("GPmt", index)
                    :
                        getDetectorPath("GPmt", index)
                    ;

    // relocate from inside the "digested" idpath up to eg export/Dayabay/GPmt/0
    // as analytic PMT definitions dont change with the geometry selection parameters

}

std::string OpticksResource::getObjectPath(const char* name, unsigned int index)
{
    assert(m_idpath && "OpticksResource::getObjectPath idpath not set");
    fs::path dir(m_idpath);
    dir /= name ;
    dir /= boost::lexical_cast<std::string>(index) ;
    return dir.string() ;
}


std::string OpticksResource::getRelativePath(const char* name, unsigned int index)
{
    // used eg by GPmt::loadFromCache returning "GPmt/0"
    fs::path reldir(name);
    reldir /= boost::lexical_cast<std::string>(index) ;
    return reldir.string() ;
}


std::string OpticksResource::getDetectorPath(const char* name, unsigned int index)
{
    assert(m_detector_base && "OpticksResource::getDetectorPath detector_path not set");
    fs::path dir(m_detector_base);
    dir /= name ;
    dir /= boost::lexical_cast<std::string>(index) ;
    return dir.string() ;
}



std::string OpticksResource::getPropertyLibDir(const char* name)
{
    assert(m_idpath && "OpticksResource::getPropertyLibDir idpath not set");
    fs::path cachedir(m_idpath);
    fs::path pld(cachedir/name );
    return pld.string() ;
}

std::string OpticksResource::getPreferenceDir(const char* type, const char* udet, const char* subtype )
{
    fs::path prefdir(PREFERENCE_BASE) ;
    if(udet) prefdir /= udet ;
    prefdir /= type ; 
    if(subtype) prefdir /= subtype ; 
    return prefdir.string() ;
}






bool OpticksResource::loadPreference(std::map<std::string, std::string>& mss, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    typedef Map<std::string, std::string> MSS ;  
    MSS* pref = MSS::load(prefdir.c_str(), name ) ; 
    if(pref)
        mss = pref->getMap(); 
    return pref != NULL ; 
}

bool OpticksResource::loadPreference(std::map<std::string, unsigned int>& msu, const char* type, const char* name)
{
    std::string prefdir = getPreferenceDir(type);
    typedef Map<std::string, unsigned int> MSU ;  
    MSU* pref = MSU::load(prefdir.c_str(), name ) ; 
    if(pref)
        msu = pref->getMap(); 
    return pref != NULL ; 
}

bool OpticksResource::existsFile(const char* path)
{
    fs::path fpath(path);
    return fs::exists(fpath ) && fs::is_regular_file(fpath) ;
}

bool OpticksResource::existsFile(const char* dir, const char* name)
{
    fs::path fpath(dir);
    fpath /= name ; 
    return fs::exists(fpath ) && fs::is_regular_file(fpath) ;
}

bool OpticksResource::existsDir(const char* path)
{
    fs::path fpath(path);
    return fs::exists(fpath ) && fs::is_directory(fpath) ;
}




std::string OpticksResource::makeSidecarPath(const char* path, const char* styp, const char* dtyp)
{
   std::string empty ; 

   fs::path src(path);
   fs::path ext = src.extension();
   bool is_styp = ext.string().compare(styp) == 0  ;

   assert(is_styp && "OpticksResource::makeSidecarPath source file type doesnt match the path file type");

   fs::path dst(path);
   dst.replace_extension(dtyp) ;

   /*
   LOG(debug) << "OpticksResource::makeSidecarPath"
             << " styp " << styp
             << " dtyp " << dtyp
             << " ext "  << ext 
             << " src " << src.string()
             << " dst " << dst.string()
             << " is_styp "  << is_styp 
             ;
   */

   return dst.string() ;
}

bool OpticksResource::loadMetadata(std::map<std::string, std::string>& mdd, const char* path)
{
    typedef Map<std::string, std::string> MSS ;  
    MSS* meta = MSS::load(path) ; 
    if(meta)
        mdd = meta->getMap(); 
    return meta != NULL ; 
}

void OpticksResource::dumpMetadata(std::map<std::string, std::string>& mdd)
{
    typedef std::map<std::string, std::string> SS ;
    for(SS::const_iterator it=mdd.begin() ; it != mdd.end() ; it++)
    {
       std::cout
             << std::setw(20) << it->first 
             << std::setw(20) << it->second
             << std::endl ; 
    }
}


bool OpticksResource::hasMetaKey(const char* key)
{
    return m_metadata.count(key) == 1 ; 
}
const char* OpticksResource::getMetaValue(const char* key)
{
    return m_metadata.count(key) == 1 ? m_metadata[key].c_str() : NULL ;
}


OpticksColors* OpticksResource::getColors()
{
    if(!m_colors)
    {
        // deferred to avoid output prior to logging setup
        std::string prefdir = getPreferenceDir("GCache");
        m_colors = OpticksColors::load(prefdir.c_str(),"GColors.json");   // colorname => hexcode
    }
    return m_colors ;
}

OpticksFlags* OpticksResource::getFlags()
{
    if(!m_flags)
    {
        //m_flags = new OpticksFlags(m_opticks); 
        m_flags = new OpticksFlags(); 
        m_flags->save(getIdPath());
    }
    return m_flags ;
}


OpticksAttrSeq* OpticksResource::getFlagNames()
{
    if(!m_flagnames)
    {
        OpticksFlags* flags = getFlags();
        Index* index = flags->getIndex();

        m_flagnames = new OpticksAttrSeq(m_opticks, "GFlags");
        m_flagnames->loadPrefs(); // color, abbrev and order 
        m_flagnames->setSequence(index);
        m_flagnames->setCtrl(OpticksAttrSeq::SEQUENCE_DEFAULTS);    
    }
    return m_flagnames ; 
}


std::map<unsigned int, std::string> OpticksResource::getFlagNamesMap()
{
    OpticksAttrSeq* flagnames = getFlagNames();
    return flagnames->getNamesMap() ; 
}








Typ* OpticksResource::getTyp()
{
    if(m_typ == NULL)
    {   
       m_typ = new Typ ; 
    }   
    return m_typ ; 
}


Types* OpticksResource::getTypes()
{
    if(!m_types)
    {   
        // deferred because idpath not known at init ?
        m_types = new Types ;   
        m_types->saveFlags(getIdPath(), ".ini");
    }   
    return m_types ;
}


