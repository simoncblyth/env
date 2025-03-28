#pragma once

#include <map>
#include <string>

//template <typename T> class NPY ; 
#include "NPY.hpp"

class Timer ; 
class Parameters ;
class Report ;
class TimesTable ; 


class Index ; 
class ViewNPY ;
class MultiViewNPY ;
class RecordsNPY ; 
class PhotonsNPY ; 
class NPYSpec ; 

class OpticksDomain ; 

/*
OpticksEvent
=============


Steps : G4 Only
-----------------

nopstep
      non-optical steps
genstep
      scintillation or cerenkov


Steps : G4 or Op
--------------------

records
      photon step records
photons
      last photon step at absorption, detection
sequence   
      photon level material/flag histories


Other : G4 or Op Indexing
--------------------------

phosel
      obtained by indexing *sequence*
recsel
      obtained by repeating *phosel* by maxrec


Not currently Used
-------------------

incoming
      slated for NumpyServer revival
aux
      was formerly used for photon level debugging 
primary
      hold initial particle info



*/

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksEvent {
      //friend class Opticks ; 
   public:
      static const char* PARAMETERS_NAME ;  
      static const char* TIMEFORMAT ;  
      static std::string timestamp();
   public:
      //
      //    typ: cerenkov/scintillaton/torch/g4gun
      //    tag: 1/-1/2/-2/...  convention: -ve tags propagated by Geant4, +ve by Opticks
      //    det: dayabay/...    identifes the geocache  
      //    cat: optional override of det for test categorization, eg PmtInBox
      //
      static OpticksEvent* load(const char* typ, const char* tag, const char* det, const char* cat=NULL, bool verbose=false);
      static Index* loadHistoryIndex(  const char* typ, const char* tag, const char* udet);
      static Index* loadMaterialIndex( const char* typ, const char* tag, const char* udet);
      static Index* loadBoundaryIndex( const char* typ, const char* tag, const char* udet);
      static Index* loadNamedIndex(    const char* typ, const char* tag, const char* udet, const char* name);
   public:
       OpticksEvent(const char* typ, const char* tag, const char* det, const char* cat=NULL);
       // CAUTION: typically created via Opticks::makeEvent 
       //          which sets maxrec before creating buffers
   public:
       bool isNoLoad();
       bool isLoaded();
       bool isIndexed();
       bool isStep();
       bool isFlat();
   public:
       void postPropagateGeant4(); // called following dynamic photon/record/sequence collection
   public:
       // from parameters
       unsigned int getBounceMax();
       unsigned int getRngMax();
       std::string getTimeStamp();
   private:
       void init();
       void indexPhotonsCPU();
   public:
       static const char* genstep_ ;
       static const char* nopstep_ ;
       static const char* photon_ ;
       static const char* record_  ;
       static const char* phosel_ ;
       static const char* recsel_  ;
       static const char* sequence_  ;
   public:
       NPY<float>* loadGenstepFromFile(int modulo=0);
       NPY<float>* loadGenstepDerivativeFromFile(const char* postfix="track", bool quietly=false);
       void setGenstepData(NPY<float>* genstep_data);
       void setNopstepData(NPY<float>* nopstep_data);
       void zero();
       void dumpDomains(const char* msg="OpticksEvent::dumpDomains");
   public:
       Parameters* getParameters();
       Timer*      getTimer();
       TimesTable* getTimesTable();
   public:
       void makeReport();
       void saveReport();
       void loadReport();
   private:
       void saveReport(const char* dir);
   public:
       void setMaxRec(unsigned int maxrec);         // maximum record slots per photon
   public:
       // G4 related qtys used by cfg4- when OpticksEvent used to store G4 propagations
       void setNumG4Event(unsigned int n);
       void setNumPhotonsPerG4Event(unsigned int n);
       unsigned int getNumG4Event();
       unsigned int getNumPhotonsPerG4Event();
   public:
       void setBoundaryIndex(Index* bndidx);
       void setHistoryIndex(Index* seqhis);
       void setMaterialIndex(Index* seqmat);
       Index* getBoundaryIndex();
       Index* getHistoryIndex();
       Index* getMaterialIndex();
   public:
       // domains used for record compression
       void setSpaceDomain(const glm::vec4& space_domain);
       void setTimeDomain(const glm::vec4& time_domain);
       void setWavelengthDomain(const glm::vec4& wavelength_domain);
       const glm::vec4& getSpaceDomain();
       const glm::vec4& getTimeDomain();
       const glm::vec4& getWavelengthDomain();
   private:
       void updateDomainsBuffer();
       void importDomainsBuffer();
   public:
       void save(bool verbose=false);
       void saveIndex(bool verbose=false);
       void loadIndex();
       void loadBuffers(bool verbose=true);
   public: 
       void createBuffers(); 
       void createSpec(); 
   private:
       void setPhotonData(NPY<float>* photon_data);
       void setSequenceData(NPY<unsigned long long>* history_data);
       void setRecordData(NPY<short>* record_data);
       void setRecselData(NPY<unsigned char>* recsel_data);
       void setPhoselData(NPY<unsigned char>* phosel_data);
   public:
       static std::string speciesDir(const char* species, const char* udet, const char* typ);
   private:
       void recordDigests();
       std::string getSpeciesDir(const char* species); // tag in the name
       std::string getTagDir(const char* species, bool tstamp);     // tag in the dir 
       void saveParameters();
       void loadParameters();
   public:
       void setFDomain(NPY<float>* fdom);
       void setIDomain(NPY<int>* idom);
   public:
       NPY<float>*          getGenstepData();
       NPY<float>*          getNopstepData();
       NPY<float>*          getPhotonData();
       NPY<short>*          getRecordData();
       NPY<unsigned char>*  getPhoselData();
       NPY<unsigned char>*  getRecselData();
       NPY<unsigned long long>*  getSequenceData();
   public:
       NPYBase*             getData(const char* name);
       std::string          getShapeString(); 
   public:
       // optionals lodged here for debug dumping single photons/records  
       void setRecordsNPY(RecordsNPY* recs);
       void setPhotonsNPY(PhotonsNPY* pho);
       RecordsNPY*          getRecordsNPY();
       PhotonsNPY*          getPhotonsNPY();
       NPY<float>*          getFDomain();
       NPY<int>*            getIDomain();
   public:
       void setFakeNopstepPath(const char* path);
   public:
       MultiViewNPY* getGenstepAttr();
       MultiViewNPY* getNopstepAttr();
       MultiViewNPY* getPhotonAttr();
       MultiViewNPY* getRecordAttr();
       MultiViewNPY* getPhoselAttr();
       MultiViewNPY* getRecselAttr();
       MultiViewNPY* getSequenceAttr();

       ViewNPY* operator [](const char* spec);

   public:
       unsigned int getNumGensteps();
       unsigned int getNumNopsteps();
       unsigned int getNumPhotons();
       unsigned int getNumRecords();
       unsigned int getMaxRec();  // per-photon
   private:
       // set by setGenstepData based on summation over Cerenkov/Scintillation photons to generate
       void setNumPhotons(unsigned int num_photons);
       void resize();
   public:
       void Summary(const char* msg="OpticksEvent::Summary");
       std::string  description(const char* msg="OpticksEvent::description");
       void         dumpPhotonData();
       static void  dumpPhotonData(NPY<float>* photon_data);

       const char*  getTyp();
       const char*  getTag();
       const char*  getDet();
       const char*  getCat();
       const char*  getUDet();
   private:
       const char*           m_typ ; 
       const char*           m_tag ; 
       const char*           m_det ; 
       const char*           m_cat ; 

       bool                  m_noload ; 
       bool                  m_loaded ; 

       Timer*                m_timer ;
       Parameters*           m_parameters ;
       Report*               m_report ;
       TimesTable*           m_ttable ;

       NPY<float>*           m_primary_data ; 
       NPY<float>*           m_genstep_data ;
       NPY<float>*           m_nopstep_data ;
       NPY<float>*           m_photon_data ;
       NPY<short>*           m_record_data ;
       NPY<unsigned char>*   m_phosel_data ;
       NPY<unsigned char>*   m_recsel_data ;
       NPY<unsigned long long>*  m_sequence_data ;

       OpticksDomain*        m_domain ; 

       MultiViewNPY*   m_genstep_attr ;
       MultiViewNPY*   m_nopstep_attr ;
       MultiViewNPY*   m_photon_attr  ;
       MultiViewNPY*   m_record_attr  ;
       MultiViewNPY*   m_phosel_attr  ;
       MultiViewNPY*   m_recsel_attr  ;
       MultiViewNPY*   m_sequence_attr  ;

       RecordsNPY*     m_records ; 
       PhotonsNPY*     m_photons ; 

       unsigned int    m_num_gensteps ; 
       unsigned int    m_num_nopsteps ; 
       unsigned int    m_num_photons ; 

       Index*          m_seqhis ; 
       Index*          m_seqmat ; 
       Index*          m_bndidx ; 

       std::vector<std::string>           m_data_names ; 
       std::map<std::string, std::string> m_abbrev ; 

       const char*     m_fake_nopstep_path ; 

       NPYSpec* m_fdom_spec ;  
       NPYSpec* m_idom_spec ;  
       NPYSpec* m_genstep_spec ;  
       NPYSpec* m_nopstep_spec ;  
       NPYSpec* m_photon_spec ;  
       NPYSpec* m_record_spec ;  
       NPYSpec* m_phosel_spec ;  
       NPYSpec* m_recsel_spec ;  
       NPYSpec* m_sequence_spec ;  

};

//
// avoiding class members simplifies usage, as full type spec is not needed for pointers : forward declarations sufficient
// for this reason moved glm domain vector members down into OpticksDomain
// this simplifies use with nvcc compiler
// 

#include "OKCORE_TAIL.hh"
  
