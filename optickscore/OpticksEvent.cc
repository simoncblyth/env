
#ifdef _MSC_VER
// 'ViewNPY': object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif


#include <climits>
#include <cassert>
#include <sstream>
#include <cstring>

// brap-
#include "BStr.hh"
#include "BTime.hh"

// npy-
#include "uif.h"
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NPYSpec.hpp"

#include "G4StepNPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "Parameters.hpp"
#include "GLMFormat.hpp"
#include "Index.hpp"

#include "Report.hpp"
#include "Timer.hpp"
#include "Times.hpp"
#include "TimesTable.hpp"

// okc-
#include "OpticksConst.hh"
#include "OpticksDomain.hh"
#include "OpticksEvent.hh"
#include "Indexer.hh"

#include "PLOG.hh"


#define TIMER(s) \
    { \
       if(m_timer)\
       {\
          Timer& t = *(m_timer) ;\
          t((s)) ;\
       }\
    }





const char* OpticksEvent::TIMEFORMAT = "%Y%m%d_%H%M%S" ;
const char* OpticksEvent::PARAMETERS_NAME = "parameters.json" ;


std::string OpticksEvent::timestamp()
{
    std::string timestamp = BTime::now(TIMEFORMAT, 0);
    return timestamp ; 
}

const char* OpticksEvent::genstep_ = "genstep" ; 
const char* OpticksEvent::nopstep_ = "nopstep" ; 
const char* OpticksEvent::photon_  = "photon" ; 
const char* OpticksEvent::record_  = "record" ; 
const char* OpticksEvent::phosel_ = "phosel" ; 
const char* OpticksEvent::recsel_  = "recsel" ; 
const char* OpticksEvent::sequence_  = "sequence" ; 


OpticksEvent::OpticksEvent(const char* typ, const char* tag, const char* det, const char* cat) 
          :
          m_typ(strdup(typ)),
          m_tag(strdup(tag)),
          m_det(strdup(det)),
          m_cat(cat ? strdup(cat) : NULL),

          m_noload(false),
          m_loaded(false),

          m_timer(NULL),
          m_parameters(NULL),
          m_report(NULL),
          m_ttable(NULL),

          m_primary_data(NULL),
          m_genstep_data(NULL),
          m_nopstep_data(NULL),
          m_photon_data(NULL),
          m_record_data(NULL),
          m_phosel_data(NULL),
          m_recsel_data(NULL),
          m_sequence_data(NULL),

          m_domain(NULL),

          m_genstep_attr(NULL),
          m_nopstep_attr(NULL),
          m_photon_attr(NULL),
          m_record_attr(NULL),
          m_phosel_attr(NULL),
          m_recsel_attr(NULL),
          m_sequence_attr(NULL),

          m_records(NULL),
          m_photons(NULL),
          m_num_gensteps(0),
          m_num_nopsteps(0),
          m_num_photons(0),

          m_seqhis(NULL),
          m_seqmat(NULL),
          m_bndidx(NULL),
          m_fake_nopstep_path(NULL),

          m_fdom_spec(NULL),
          m_idom_spec(NULL),
          m_genstep_spec(NULL),
          m_nopstep_spec(NULL),
          m_photon_spec(NULL),
          m_record_spec(NULL),
          m_phosel_spec(NULL),
          m_recsel_spec(NULL),
          m_sequence_spec(NULL)
{
    init();
}


bool OpticksEvent::isNoLoad()
{
    return m_noload ; 
}
bool OpticksEvent::isLoaded()
{
    return m_loaded ; 
}
bool OpticksEvent::isStep()
{
    return true  ; 
}
bool OpticksEvent::isFlat()
{
    return false  ; 
}




unsigned int OpticksEvent::getNumGensteps()
{
    return m_num_gensteps ; 
}
unsigned int OpticksEvent::getNumNopsteps()
{
    return m_num_nopsteps ; 
}

void OpticksEvent::setNumPhotons(unsigned int num_photons)
{
    m_num_photons = num_photons ; 
    resize();
}
unsigned int OpticksEvent::getNumPhotons()
{
    return m_num_photons ; 
}


unsigned int OpticksEvent::getNumRecords()
{
    unsigned int maxrec = getMaxRec();
    return m_num_photons * maxrec ; 
}
unsigned int OpticksEvent::getMaxRec()
{
    return m_domain->getMaxRec() ; 
}
void OpticksEvent::setMaxRec(unsigned int maxrec)
{
    m_domain->setMaxRec(maxrec);
}





NPY<float>* OpticksEvent::getGenstepData(){ return m_genstep_data ; }
NPY<float>* OpticksEvent::getNopstepData() { return m_nopstep_data ; }
NPY<float>* OpticksEvent::getPhotonData(){ return m_photon_data ; } 
NPY<short>* OpticksEvent::getRecordData(){ return m_record_data ; }
NPY<unsigned char>* OpticksEvent::getPhoselData(){ return m_phosel_data ; }
NPY<unsigned char>* OpticksEvent::getRecselData(){ return m_recsel_data ; }
NPY<unsigned long long>* OpticksEvent::getSequenceData(){ return m_sequence_data ; }

MultiViewNPY* OpticksEvent::getGenstepAttr(){ return m_genstep_attr ; }
MultiViewNPY* OpticksEvent::getNopstepAttr(){ return m_nopstep_attr ; }
MultiViewNPY* OpticksEvent::getPhotonAttr(){ return m_photon_attr ; }
MultiViewNPY* OpticksEvent::getRecordAttr(){ return m_record_attr ; }
MultiViewNPY* OpticksEvent::getPhoselAttr(){ return m_phosel_attr ; }
MultiViewNPY* OpticksEvent::getRecselAttr(){ return m_recsel_attr ; }
MultiViewNPY* OpticksEvent::getSequenceAttr(){ return m_sequence_attr ; }



void OpticksEvent::setRecordsNPY(RecordsNPY* records)
{
    m_records = records ; 
}
RecordsNPY* OpticksEvent::getRecordsNPY()
{
    return m_records ;
}

void OpticksEvent::setPhotonsNPY(PhotonsNPY* photons)
{
    m_photons = photons ; 
}
PhotonsNPY* OpticksEvent::getPhotonsNPY()
{
    return m_photons ;
}


const char* OpticksEvent::getTyp()
{
    return m_typ ; 
}
const char* OpticksEvent::getTag()
{
    return m_tag ; 
}
const char* OpticksEvent::getDet()
{
    return m_det ; 
}
const char* OpticksEvent::getCat()
{
    return m_cat ; 
}
const char* OpticksEvent::getUDet()
{
    return m_cat && strlen(m_cat) > 0 ? m_cat : m_det ; 
}









void OpticksEvent::setFDomain(NPY<float>* fdom)
{
    m_domain->setFDomain(fdom) ; 
}
void OpticksEvent::setIDomain(NPY<int>* idom)
{
    m_domain->setIDomain(idom) ; 
}

NPY<float>* OpticksEvent::getFDomain()
{
    return m_domain->getFDomain() ; 
}
NPY<int>* OpticksEvent::getIDomain()
{
    return m_domain->getIDomain() ; 
}

void OpticksEvent::setSpaceDomain(const glm::vec4& space_domain)
{
    m_domain->setSpaceDomain(space_domain) ; 
}
void OpticksEvent::setTimeDomain(const glm::vec4& time_domain)
{
    m_domain->setTimeDomain(time_domain)  ; 
}
void OpticksEvent::setWavelengthDomain(const glm::vec4& wavelength_domain)
{
    m_domain->setWavelengthDomain(wavelength_domain)  ; 
}


const glm::vec4& OpticksEvent::getSpaceDomain()
{
    return m_domain->getSpaceDomain() ; 
}
const glm::vec4& OpticksEvent::getTimeDomain()
{
    return m_domain->getTimeDomain() ;
}
const glm::vec4& OpticksEvent::getWavelengthDomain()
{ 
    return m_domain->getWavelengthDomain() ; 
}





void OpticksEvent::setBoundaryIndex(Index* bndidx)
{
    // called from OpIndexer::indexBoundaries
    m_bndidx = bndidx ; 
}
void OpticksEvent::setHistoryIndex(Index* seqhis)
{
    // called from OpIndexer::indexSequenceLoaded 
    m_seqhis = seqhis ; 
}
void OpticksEvent::setMaterialIndex(Index* seqmat)
{
    // called from OpIndexer::indexSequenceLoaded
    m_seqmat = seqmat ; 
}


Index* OpticksEvent::getHistoryIndex()
{
    return m_seqhis ; 
} 
Index* OpticksEvent::getMaterialIndex()
{
    return m_seqmat ; 
} 
Index* OpticksEvent::getBoundaryIndex()
{
    return m_bndidx ; 
}




Parameters* OpticksEvent::getParameters()
{
    return m_parameters ;
}
Timer* OpticksEvent::getTimer()
{
    return m_timer ;
}
TimesTable* OpticksEvent::getTimesTable()
{
    return m_ttable ;
}






void OpticksEvent::init()
{
    m_timer = new Timer("OpticksEvent"); 
    m_timer->setVerbose(false);
    m_timer->start();

    m_parameters = new Parameters ;
    m_report = new Report ; 
    m_domain = new OpticksDomain ; 

    m_parameters->add<std::string>("TimeStamp", timestamp() );
    m_parameters->add<std::string>("Type", m_typ );
    m_parameters->add<std::string>("Tag", m_tag );
    m_parameters->add<std::string>("Detector", m_det );
    if(m_cat) m_parameters->add<std::string>("Cat", m_cat );
    m_parameters->add<std::string>("UDet", getUDet() );

    m_data_names.push_back(genstep_);
    m_data_names.push_back(nopstep_);
    m_data_names.push_back(photon_);
    m_data_names.push_back(record_);
    m_data_names.push_back(phosel_);
    m_data_names.push_back(recsel_);
    m_data_names.push_back(sequence_);

    m_abbrev[genstep_] = "" ;      // no-prefix : cerenkov or scintillation
    m_abbrev[nopstep_] = "no" ;    // non optical particle steps obtained from G4 eg with g4gun
    m_abbrev[photon_] = "ox" ;     // photon final step uncompressed 
    m_abbrev[record_] = "rx" ;     // photon step compressed record
    m_abbrev[phosel_] = "ps" ;     // photon selection index
    m_abbrev[recsel_] = "rs" ;     // record selection index
    m_abbrev[sequence_] = "ph" ;   // (unsigned long long) photon seqhis/seqmat
}


NPYBase* OpticksEvent::getData(const char* name)
{
    NPYBase* data = NULL ; 
    if(     strcmp(name, genstep_)==0) data = static_cast<NPYBase*>(m_genstep_data) ; 
    else if(strcmp(name, nopstep_)==0) data = static_cast<NPYBase*>(m_nopstep_data) ;
    else if(strcmp(name, photon_)==0)  data = static_cast<NPYBase*>(m_photon_data) ;
    else if(strcmp(name, record_)==0)  data = static_cast<NPYBase*>(m_record_data) ;
    else if(strcmp(name, phosel_)==0)  data = static_cast<NPYBase*>(m_phosel_data) ;
    else if(strcmp(name, recsel_)==0)  data = static_cast<NPYBase*>(m_recsel_data) ;
    else if(strcmp(name, sequence_)==0) data = static_cast<NPYBase*>(m_sequence_data) ;
    return data ; 
}

std::string OpticksEvent::getShapeString()
{
    std::stringstream ss ; 
    for(std::vector<std::string>::const_iterator it=m_data_names.begin() ; it != m_data_names.end() ; it++)
    {
         std::string name = *it ; 
         NPYBase* data = getData(name.c_str());
         ss << " " << name << " " << ( data ? data->getShapeString() : "NULL" )  ; 
    }
    return ss.str();
}

std::string OpticksEvent::getTimeStamp()
{
    return m_parameters->get<std::string>("TimeStamp");
}
unsigned int OpticksEvent::getBounceMax()
{
    return m_parameters->get<unsigned int>("BounceMax");
}
unsigned int OpticksEvent::getRngMax()
{
    return m_parameters->get<unsigned int>("RngMax");
}


ViewNPY* OpticksEvent::operator [](const char* spec)
{
    std::vector<std::string> elem ; 
    BStr::split(elem, spec, '.');

    if(elem.size() != 2 ) assert(0);

    MultiViewNPY* mvn(NULL); 
    if(     elem[0] == genstep_)  mvn = m_genstep_attr ;  
    else if(elem[0] == nopstep_)  mvn = m_nopstep_attr ;
    else if(elem[0] == photon_)   mvn = m_photon_attr ;
    else if(elem[0] == record_)   mvn = m_record_attr ;
    else if(elem[0] == phosel_)   mvn = m_phosel_attr ;
    else if(elem[0] == recsel_)   mvn = m_recsel_attr ;
    else if(elem[0] == sequence_) mvn = m_sequence_attr ;

    assert(mvn);
    return (*mvn)[elem[1].c_str()] ;
}

void OpticksEvent::createSpec()
{
    // invoked by Opticks::makeEvent   or OpticksEvent::load
    unsigned int maxrec = getMaxRec();

    m_genstep_spec = new NPYSpec(0,6,4,0, NPYBase::FLOAT) ;
    m_fdom_spec = new NPYSpec(3,1,4,0, NPYBase::FLOAT) ;
    m_idom_spec = new NPYSpec(1,1,4,0, NPYBase::INT) ;

    m_nopstep_spec = new NPYSpec(0,4,4,0, NPYBase::FLOAT) ;
    m_photon_spec = new NPYSpec(0,4,4,0, NPYBase::FLOAT) ;
    m_sequence_spec = new NPYSpec(0,1,2,0, NPYBase::ULONGLONG) ;
    m_phosel_spec = new NPYSpec(0,1,4,0, NPYBase::UCHAR) ;
    m_record_spec = new NPYSpec(0,maxrec,2,4, NPYBase::SHORT) ;
    m_recsel_spec = new NPYSpec(0,maxrec,1,4, NPYBase::UCHAR) ;
}


void OpticksEvent::createBuffers()
{
    // invoked by Opticks::makeEvent 

    // NB allocation is deferred until zeroing and they start at 0 items anyhow
    // NB no gensteps yet, those come externally 

    NPY<float>* nop = NPY<float>::make(m_nopstep_spec); 
    setNopstepData(nop);   

    NPY<float>* pho = NPY<float>::make(m_photon_spec); // must match GPU side photon.h:PNUMQUAD
    setPhotonData(pho);   

    NPY<unsigned long long>* seq = NPY<unsigned long long>::make(m_sequence_spec);  
    setSequenceData(seq);   

    NPY<unsigned char>* phosel = NPY<unsigned char>::make(m_phosel_spec); 
    setPhoselData(phosel);   

    NPY<unsigned char>* recsel = NPY<unsigned char>::make(m_recsel_spec); 
    setRecselData(recsel);   

    NPY<short>* rec = NPY<short>::make(m_record_spec); 
    setRecordData(rec);   


    NPY<float>* fdom = NPY<float>::make(m_fdom_spec);
    setFDomain(fdom);

    NPY<int>* idom = NPY<int>::make(m_idom_spec);
    setIDomain(idom);

    // these small ones can be zeroed directly 
    fdom->zero();
    idom->zero();
}


void OpticksEvent::resize()
{
    // NB these are all photon level qtys on the first dimension
    //    including recsel and record thanks to structured arrays (num_photons, maxrec, ...)

    assert(m_photon_data);
    assert(m_sequence_data);
    assert(m_phosel_data);
    assert(m_recsel_data);
    assert(m_record_data);

    unsigned int num_photons = getNumPhotons();
    unsigned int num_records = getNumRecords();
    unsigned int maxrec = getMaxRec();
 

    LOG(info) << "OpticksEvent::resize " 
              << " num_photons " << num_photons  
              << " num_records " << num_records 
              << " maxrec " << maxrec
              ;

    m_photon_data->setNumItems(num_photons);
    m_sequence_data->setNumItems(num_photons);
    m_phosel_data->setNumItems(num_photons);
    m_recsel_data->setNumItems(num_photons);
    m_record_data->setNumItems(num_photons);

    m_parameters->add<unsigned int>("NumGensteps", getNumGensteps());
    m_parameters->add<unsigned int>("NumPhotons",  getNumPhotons());
    m_parameters->add<unsigned int>("NumRecords",  getNumRecords());

}


void OpticksEvent::zero()
{
    if(m_photon_data)   m_photon_data->zero();
    if(m_sequence_data) m_sequence_data->zero();
    if(m_record_data)   m_record_data->zero();

    // when operating CPU side phosel and recsel are derived from sequence data
    // when operating GPU side they need not ever come to CPU
    //if(m_phosel_data)   m_phosel_data->zero();
    //if(m_recsel_data)   m_recsel_data->zero();
}


void OpticksEvent::dumpDomains(const char* msg)
{
    m_domain->dump(msg);
}
void OpticksEvent::updateDomainsBuffer()
{
    m_domain->updateBuffer();
}
void OpticksEvent::importDomainsBuffer()
{
    m_domain->importBuffer();
}



void OpticksEvent::setGenstepData(NPY<float>* genstep)
{
    m_genstep_data = genstep  ;
    m_parameters->add<std::string>("genstepDigest",   genstep->getDigestString()  );

    //                                                j k l sz   type        norm   iatt  item_from_dim
    ViewNPY* vpos = new ViewNPY("vpos",m_genstep_data,1,0,0,4,ViewNPY::FLOAT,false,false, 1);    // (x0, t0)                     2nd GenStep quad 
    ViewNPY* vdir = new ViewNPY("vdir",m_genstep_data,2,0,0,4,ViewNPY::FLOAT,false,false, 1);    // (DeltaPosition, step_length) 3rd GenStep quad

    m_genstep_attr = new MultiViewNPY("genstep_attr");
    m_genstep_attr->add(vpos);
    m_genstep_attr->add(vdir);

    {
        m_num_gensteps = m_genstep_data->getShape(0) ;
        unsigned int num_photons = m_genstep_data->getUSum(0,3);
        setNumPhotons(num_photons); // triggers a resize   <<<<<<<<<<<<< SPECIAL HANDLING OF GENSTEP <<<<<<<<<<<<<<
    }
}

void OpticksEvent::setPhotonData(NPY<float>* photon_data)
{
    m_photon_data = photon_data  ;
    if(m_num_photons == 0) 
    {
        m_num_photons = photon_data->getShape(0) ;

        LOG(debug) << "OpticksEvent::setPhotonData"
                  << " setting m_num_photons from shape(0) " << m_num_photons 
                  ;
    }
    else
    {
        assert(m_num_photons == photon_data->getShape(0));
    }

    m_photon_data->setDynamic();  // need to update with seeding so GL_DYNAMIC_DRAW needed 
    m_photon_attr = new MultiViewNPY("photon_attr");
    //                                                  j k l,sz   type          norm   iatt  item_from_dim
    m_photon_attr->add(new ViewNPY("vpos",m_photon_data,0,0,0,4,ViewNPY::FLOAT, false, false, 1));      // 1st quad
    m_photon_attr->add(new ViewNPY("vdir",m_photon_data,1,0,0,4,ViewNPY::FLOAT, false, false, 1));      // 2nd quad
    m_photon_attr->add(new ViewNPY("vpol",m_photon_data,2,0,0,4,ViewNPY::FLOAT, false, false, 1));      // 3rd quad
    m_photon_attr->add(new ViewNPY("iflg",m_photon_data,3,0,0,4,ViewNPY::INT  , false, true , 1));      // 4th quad

    //
    //  photon array 
    //  ~~~~~~~~~~~~~
    //     
    //  vpos  xxxx yyyy zzzz wwww    position, time           [:,0,:4]
    //  vdir  xxxx yyyy zzzz wwww    direction, wavelength    [:,1,:4]
    //  vpol  xxxx yyyy zzzz wwww    polarization weight      [:,2,:4] 
    //  iflg  xxxx yyyy zzzz wwww                             [:,3,:4]
    //
    //
    //  record array
    //  ~~~~~~~~~~~~~~
    //       
    //              4*short(snorm)
    //          ________
    //  rpos    xxyyzzww 
    //  rpol->  xyzwaabb <-rflg 
    //          ----^^^^
    //     4*ubyte     2*ushort   
    //     (unorm)     (iatt)
    //
    //
    //
    // corresponds to GPU side cu/photon.h:psave and rsave 
    //
}



void OpticksEvent::setNopstepData(NPY<float>* nopstep)
{
    m_nopstep_data = nopstep  ;
    if(!nopstep) return ; 

    m_num_nopsteps = m_nopstep_data->getShape(0) ;
    LOG(debug) << "OpticksEvent::setNopstepData"
              << " shape " << nopstep->getShapeString()
              ;

    //                                                j k l sz   type         norm   iatt   item_from_dim
    ViewNPY* vpos = new ViewNPY("vpos",m_nopstep_data,0,0,0,4,ViewNPY::FLOAT ,false,  false, 1);
    ViewNPY* vdir = new ViewNPY("vdir",m_nopstep_data,1,0,0,4,ViewNPY::FLOAT ,false,  false, 1);   
    ViewNPY* vpol = new ViewNPY("vpol",m_nopstep_data,2,0,0,4,ViewNPY::FLOAT ,false,  false, 1);   

    m_nopstep_attr = new MultiViewNPY("nopstep_attr");
    m_nopstep_attr->add(vpos);
    m_nopstep_attr->add(vdir);
    m_nopstep_attr->add(vpol);

}


void OpticksEvent::setRecordData(NPY<short>* record_data)
{
    m_record_data = record_data  ;


#ifdef OLDWAY
    //                                               j k l  sz   type                  norm   iatt   item_from_dim
    ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,0 ,4,ViewNPY::SHORT          ,true,  false, 2);
    ViewNPY* rpol = new ViewNPY("rpol",m_record_data,1,0,0 ,4,ViewNPY::UNSIGNED_BYTE  ,true,  false, 2);   

    ViewNPY* rflg = new ViewNPY("rflg",m_record_data,1,2,0 ,2,ViewNPY::UNSIGNED_SHORT ,false, true,  2);   
    // NB k=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  

    ViewNPY* rflq = new ViewNPY("rflq",m_record_data,1,2,0 ,4,ViewNPY::UNSIGNED_BYTE  ,false, true,  2);   
    // NB k=2 again : try a UBYTE view of the same data for access to boundary,m1,history-hi,history-lo

#else
    // see ggv-/issues/gui_broken_photon_record_colors.rst note the shift of one to the right of the (j,k,l)

    //                                               j k l  sz   type                  norm   iatt   item_from_dim
    ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,0 ,4,ViewNPY::SHORT          ,true,  false, 2);
    ViewNPY* rpol = new ViewNPY("rpol",m_record_data,0,1,0 ,4,ViewNPY::UNSIGNED_BYTE  ,true,  false, 2);   

    ViewNPY* rflg = new ViewNPY("rflg",m_record_data,0,1,2 ,2,ViewNPY::UNSIGNED_SHORT ,false, true,  2);   
    // NB k=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  

    ViewNPY* rflq = new ViewNPY("rflq",m_record_data,0,1,2 ,4,ViewNPY::UNSIGNED_BYTE  ,false, true,  2);   
    // NB k=2 again : try a UBYTE view of the same data for access to boundary,m1,history-hi,history-lo

#endif

    // structured record array => item_from_dim=2 the count comes from product of 1st two dimensions

    // ViewNPY::TYPE need not match the NPY<T>,
    // OpenGL shaders will view the data as of the ViewNPY::TYPE, 
    // informed via glVertexAttribPointer/glVertexAttribIPointer 
    // in oglrap-/Rdr::address(ViewNPY* vnpy)
 
    // standard byte offsets obtained from from sizeof(T)*value_offset 
    //rpol->setCustomOffset(sizeof(unsigned char)*rpol->getValueOffset());
    // this is not needed

    m_record_attr = new MultiViewNPY("record_attr");

    m_record_attr->add(rpos);
    m_record_attr->add(rpol);
    m_record_attr->add(rflg);
    m_record_attr->add(rflq);
}


void OpticksEvent::setPhoselData(NPY<unsigned char>* phosel_data)
{
    m_phosel_data = phosel_data ;
    if(!m_phosel_data) return ; 

    //                                               j k l sz   type                norm   iatt   item_from_dim
    ViewNPY* psel = new ViewNPY("psel",m_phosel_data,0,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true, 1);
    m_phosel_attr = new MultiViewNPY("phosel_attr");
    m_phosel_attr->add(psel);
}


void OpticksEvent::setRecselData(NPY<unsigned char>* recsel_data)
{
    m_recsel_data = recsel_data ;

    if(!m_recsel_data) return ; 
    //                                               j k l sz   type                norm   iatt   item_from_dim
    ViewNPY* rsel = new ViewNPY("rsel",m_recsel_data,0,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true, 2);
    // structured recsel array, means the count needs to come from product of 1st two dimensions, 

    m_recsel_attr = new MultiViewNPY("recsel_attr");
    m_recsel_attr->add(rsel);
}


void OpticksEvent::setSequenceData(NPY<unsigned long long>* sequence_data)
{
    m_sequence_data = sequence_data  ;
    assert(sizeof(unsigned long long) == 4*sizeof(unsigned short));  
    //
    // 64 bit uint used to hold the sequence flag sequence 
    // is presented to OpenGL shaders as 4 *16bit ushort 
    // as intend to reuse the sequence bit space for the indices and count 
    // via some diddling 
    //
    //      Have not taken the diddling route, 
    //      instead using separate Recsel/Phosel buffers for the indices
    // 
    //                                                 j k l sz   type                norm   iatt    item_from_dim
    ViewNPY* phis = new ViewNPY("phis",m_sequence_data,0,0,0,4,ViewNPY::UNSIGNED_SHORT,false,  true, 1);
    ViewNPY* pmat = new ViewNPY("pmat",m_sequence_data,0,1,0,4,ViewNPY::UNSIGNED_SHORT,false,  true, 1);
    m_sequence_attr = new MultiViewNPY("sequence_attr");
    m_sequence_attr->add(phis);
    m_sequence_attr->add(pmat);

}






void OpticksEvent::dumpPhotonData()
{
    if(!m_photon_data) return ;
    dumpPhotonData(m_photon_data);
}

void OpticksEvent::dumpPhotonData(NPY<float>* photons)
{
    std::cout << photons->description("OpticksEvent::dumpPhotonData") << std::endl ;

    for(unsigned int i=0 ; i < photons->getShape(0) ; i++)
    {
        if(i%10000 == 0)
        {
            unsigned int ux = photons->getUInt(i,0,0); 
            float fx = photons->getFloat(i,0,0); 
            float fy = photons->getFloat(i,0,1); 
            float fz = photons->getFloat(i,0,2); 
            float fw = photons->getFloat(i,0,3); 
            printf(" ph  %7u   ux %7u   fxyzw %10.3f %10.3f %10.3f %10.3f \n", i, ux, fx, fy, fz, fw );             
        }
    }  
}



void OpticksEvent::Summary(const char* msg)
{
    LOG(info) << description(msg) ; 
}

std::string OpticksEvent::description(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << " " 
       << " typ: " << m_typ 
       << " tag: " << m_tag 
       << " det: " << m_det 
       << " cat: " << ( m_cat ? m_cat : "NULL" ) 
       << " udet: " << getUDet()
       << " num_photons: " <<  m_num_photons
       ;

    //if(m_genstep_data)  ss << m_genstep_data->description("m_genstep_data") ;
    //if(m_photon_data)   ss << m_photon_data->description("m_photon_data") ;

    return ss.str();
}


void OpticksEvent::recordDigests()
{
    NPY<float>* ox = getPhotonData() ;
    if(ox && ox->hasData())
        m_parameters->add<std::string>("photonData",   ox->getDigestString()  );

    NPY<short>* rx = getRecordData() ;
    if(rx && rx->hasData())
        m_parameters->add<std::string>("recordData",   rx->getDigestString()  );

    NPY<unsigned long long>* ph = getSequenceData() ;
    if(ph && ph->hasData())
        m_parameters->add<std::string>("sequenceData", ph->getDigestString()  );
}

void OpticksEvent::save(bool verbose)
{
    (*m_timer)("_save");

    recordDigests();

    const char* udet = getUDet();
    LOG(info) << "OpticksEvent::save"
              << " typ: " << m_typ
              << " tag: " << m_tag
              << " det: " << m_det
              << " cat: " << ( m_cat ? m_cat : "NULL" )
              << " udet: " << udet 
              ;    

    LOG(info) << "OpticksEvent::save " << getShapeString() ; 

   // genstep normally not saved as it exists already coming from elsewhere,
   // but for TorchStep that insnt the case


    NPY<float>* no = getNopstepData();
    //if(no)
    {
        no->setVerbose(verbose);
        no->save("no%s", m_typ,  m_tag, udet);
        no->dump("OpticksEvent::save (nopstep)");
    }

    NPY<float>* ox = getPhotonData();
    //if(ox)
    {
        ox->setVerbose(verbose);
        ox->save("ox%s", m_typ,  m_tag, udet);
    } 

    NPY<short>* rx = getRecordData();    
    //if(rx)
    {
        rx->setVerbose(verbose);
        rx->save("rx%s", m_typ,  m_tag, udet);
    }

    NPY<unsigned long long>* ph = getSequenceData();
    //if(ph)
    {
        ph->setVerbose(verbose);
        ph->save("ph%s", m_typ,  m_tag, udet);
    }




    updateDomainsBuffer();

    NPY<float>* fdom = getFDomain();
    if(fdom) fdom->save("fdom%s", m_typ,  m_tag, udet);

    NPY<int>* idom = getIDomain();
    if(idom) idom->save("idom%s", m_typ,  m_tag, udet);

    if(no)
    {
       assert(idom && "OpticksEvent::save non-null nopstep BUT HAS NULL IDOM ");
    }


    saveIndex(verbose);
    saveParameters();

    (*m_timer)("save");

    makeReport();  // after timer save, in order to include that in the report
    saveReport();
}



void OpticksEvent::makeReport()
{
    LOG(info) << "OpticksEvent::makeReport" ; 

    m_parameters->dump();

    m_timer->stop();

    m_ttable = m_timer->makeTable();
    m_ttable->dump("OpticksEvent::makeReport");

    m_report->add(m_parameters->getLines());
    m_report->add(m_ttable->getLines());
}


std::string OpticksEvent::speciesDir(const char* species, const char* udet, const char* typ)
{
    std::string dir = NPYBase::directory(species, typ, udet );
    return dir ; 
}

std::string OpticksEvent::getSpeciesDir(const char* species)
{
   // eg species  "ix" for indices
    const char* udet = getUDet();
    return speciesDir(species, udet, m_typ );
}

std::string OpticksEvent::getTagDir(const char* species, bool tstamp)
{
    std::stringstream ss ;
    ss << getSpeciesDir(species) << "/" << m_tag  ;
    if(tstamp) ss << "/" << getTimeStamp() ;
    return ss.str();
}


void OpticksEvent::saveParameters()
{
    std::string mddir = getTagDir("md", false);
    m_parameters->save(mddir.c_str(), PARAMETERS_NAME);

    std::string mddir_ts = getTagDir("md", true);
    m_parameters->save(mddir_ts.c_str(), PARAMETERS_NAME);
}


void OpticksEvent::loadParameters()
{
    std::string pmdir = getTagDir("md", false);
    m_parameters->load_(pmdir.c_str(), PARAMETERS_NAME );
}

void OpticksEvent::saveReport()
{
    std::string mdd = getTagDir("md", false);  
    saveReport(mdd.c_str());

    std::string mdd_ts = getTagDir("md", true);  
    saveReport(mdd_ts.c_str());
}



void OpticksEvent::saveReport(const char* dir)
{
    if(!m_ttable || !m_report) return ; 
    LOG(info) << "OpticksEvent::saveReport to " << dir  ; 

    m_ttable->save(dir);
    m_report->save(dir);  
}

void OpticksEvent::loadReport()
{
    std::string mdd = getTagDir("md", false);  
    m_ttable = Timer::loadTable(mdd.c_str());
    m_report = Report::load(mdd.c_str());
}

void OpticksEvent::setFakeNopstepPath(const char* path)
{
    // fake path used by OpticksEvent::load rather than standard one
    // see npy-/nopstep_viz_debug.py

    m_fake_nopstep_path = path ? strdup(path) : NULL ;
}


OpticksEvent* OpticksEvent::load(const char* typ, const char* tag, const char* det, const char* cat, bool verbose)
{
    LOG(info) << "OpticksEvent::load"
              << " typ " << typ
              << " tag " << tag
              << " det " << det
              << " cat " << ( cat ? cat : "NULL" )
              ;

    OpticksEvent* evt = new OpticksEvent(typ, tag, det, cat);

    evt->loadBuffers(verbose);
    if(evt->isNoLoad())
    {
         LOG(warning) << "OpticksEvent::load FAILED " ;
         delete evt ;
         evt = NULL ;
    } 
    return evt ;  
}


void OpticksEvent::loadBuffers(bool verbose)
{
    TIMER("_load");

    const char* udet = getUDet(); // cat overrides det if present 

    bool qload = true ; 

    const char* idom_tfmt = "idom%s" ;

    NPY<int>*   idom = NPY<int>::load(idom_tfmt, m_typ,  m_tag, udet, qload);
    if(!idom)
    {
        std::string dir = NPYBase::directory(idom_tfmt, m_typ, udet );


        m_noload = true ; 
        LOG(warning) << "OpticksEvent::load NO SUCH EVENT : RUN WITHOUT --load OPTION TO CREATE IT " 
                     << " typ: " << m_typ
                     << " tag: " << m_tag
                     << " det: " << m_det
                     << " cat: " << ( m_cat ? m_cat : "NULL" )
                     << " udet: " << udet 
                     << " dir " << dir    
                    ;     
        return ; 
    }

    m_loaded = true ; 

    NPY<float>* fdom = NPY<float>::load("fdom%s", m_typ,  m_tag, udet, qload );

    setIDomain(idom);
    setFDomain(fdom);

    loadReport();
    loadParameters();
    loadIndex();

    importDomainsBuffer();

    createSpec();      // domains import sets maxrec allowing spec to be created 

    assert(idom->hasShapeSpec(m_idom_spec));
    assert(fdom->hasShapeSpec(m_fdom_spec));
 


    NPY<float>* no = NULL ; 
    if(m_fake_nopstep_path)
    {
        LOG(warning) << "OpticksEvent::load using setFakeNopstepPath " << m_fake_nopstep_path ; 
        no = NPY<float>::debugload(m_fake_nopstep_path);
    }
    else
    {  
        no = NPY<float>::load("no%s", m_typ,  m_tag, udet, qload);
    }
    if(no) assert(no->hasItemSpec(m_nopstep_spec) );

    NPY<float>*              ox = NPY<float>::load("ox%s", m_typ,  m_tag, udet, qload);
    NPY<short>*              rx = NPY<short>::load("rx%s", m_typ,  m_tag, udet, qload);
    NPY<unsigned long long>* ph = NPY<unsigned long long>::load("ph%s", m_typ,  m_tag, udet, qload );
    NPY<unsigned char>*      ps = NPY<unsigned char>::load("ps%s", m_typ,  m_tag, udet, qload );
    NPY<unsigned char>*      rs = NPY<unsigned char>::load("rs%s", m_typ,  m_tag, udet, qload );

    if(ox) assert(ox->hasItemSpec(m_photon_spec) );
    if(rx) assert(rx->hasItemSpec(m_record_spec) );
    if(ph) assert(ph->hasItemSpec(m_sequence_spec) );
    if(ps) assert(ps->hasItemSpec(m_phosel_spec) );
    if(rs) assert(rs->hasItemSpec(m_recsel_spec) );


    unsigned int num_nopstep = no ? no->getShape(0) : 0 ;
    unsigned int num_photons = ox ? ox->getShape(0) : 0 ;
    unsigned int num_history = ph ? ph->getShape(0) : 0 ;
    unsigned int num_phosel  = ps ? ps->getShape(0) : 0 ;

    // either zero or matching 
    assert(num_history == 0 || num_photons == num_history );
    assert(num_phosel == 0 || num_photons == num_phosel );

    unsigned int num_records = rx ? rx->getShape(0) : 0 ;
    unsigned int num_recsel  = rs ? rs->getShape(0) : 0 ;

    assert(num_recsel == 0 || num_records == num_recsel );


    LOG(info) << "OpticksEvent::load shape(0) before reshaping "
              << " num_nopstep " << num_nopstep
              << " [ "
              << " num_photons " << num_photons
              << " num_history " << num_history
              << " num_phosel " << num_phosel 
              << " ] "
              << " [ "
              << " num_records " << num_records
              << " num_recsel " << num_recsel
              << " ] "
              ; 


    setNopstepData(no);
    setPhotonData(ox);
    setSequenceData(ph);
    setRecordData(rx);

    setPhoselData(ps);
    setRecselData(rs);

    (*m_timer)("load");


    LOG(info) << "OpticksEvent::load " << getShapeString() ; 

    if(verbose)
    {
        fdom->Summary("fdom");
        idom->Summary("idom");

        if(no) no->Summary("no");
        if(ox) ox->Summary("ox");
        if(rx) rx->Summary("rx");
        if(ph) ph->Summary("ph");
        if(ps) ps->Summary("ps");
        if(rs) rs->Summary("rs");
    }

}

bool OpticksEvent::isIndexed()
{
    return m_phosel_data != NULL && m_recsel_data != NULL && m_seqhis != NULL && m_seqmat != NULL ;
}



NPY<float>* OpticksEvent::loadGenstepDerivativeFromFile(const char* postfix, bool quietly)
{
    char tag[128];
    snprintf(tag, 128, "%s_%s", m_tag, postfix );


    if(!quietly)
    LOG(info) << "OpticksEvent::loadGenstepDerivativeFromFile  "
              << " typ " << m_typ
              << " tag " << tag
              << " det " << m_det
              ;

    NPY<float>* npy = NPY<float>::load(m_typ, tag, m_det, quietly) ;
    if(npy)
    {
        npy->dump("OpticksEvent::loadGenstepDerivativeFromFile");
    }
    else
    {
        if(!quietly)
        LOG(warning) << "OpticksEvent::loadGenstepDerivativeFromFile FAILED for postfix " << postfix ;
    }
    return npy ; 
}


NPY<float>* OpticksEvent::loadGenstepFromFile(int modulo)
{
    LOG(info) << "OpticksEvent::loadGenstepFromFile  "
              << " typ " << m_typ
              << " tag " << m_tag
              << " det " << m_det
              ;

    NPY<float>* npy = NPY<float>::load(m_typ, m_tag, m_det ) ;

    m_parameters->add<std::string>("genstepAsLoaded",   npy->getDigestString()  );

    m_parameters->add<int>("Modulo", modulo );

    if(modulo > 0)
    {
        LOG(warning) << "App::loadGenstepFromFile applying modulo scaledown " << modulo ;
        npy = NPY<float>::make_modulo(npy, modulo);
        m_parameters->add<std::string>("genstepModulo",   npy->getDigestString()  );
    }
    return npy ;
}





void OpticksEvent::setNumG4Event(unsigned int n)
{
   m_parameters->add<int>("NumG4Event", n);
}
void OpticksEvent::setNumPhotonsPerG4Event(unsigned int n)
{
   m_parameters->add<int>("NumPhotonsPerG4Event", n);
}
unsigned int OpticksEvent::getNumG4Event()
{
   return m_parameters->get<int>("NumG4Event");
}
unsigned int OpticksEvent::getNumPhotonsPerG4Event()
{
   return m_parameters->get<int>("NumPhotonsPerG4Event");
}
 
void OpticksEvent::postPropagateGeant4()
{
    unsigned int num_photons = m_photon_data->getShape(0);

    setNumPhotons(num_photons);  // triggers resize

    LOG(info) << "OpticksEvent::postPropagateGeant4" 
              << " num_photons " << num_photons
              ;

    indexPhotonsCPU();    
}

void OpticksEvent::indexPhotonsCPU()
{
    // see tests/IndexerTest

    NPY<unsigned long long>* sequence = getSequenceData();
    NPY<unsigned char>*        phosel = getPhoselData();
    NPY<unsigned char>*        recsel0 = getRecselData();

    LOG(info) << "OpticksEvent::indexPhotonsCPU" 
              << " sequence " << sequence->getShapeString()
              << " phosel "   << phosel->getShapeString()
              << " phosel.hasData "   << phosel->hasData()
              << " recsel0 "   << recsel0->getShapeString()
              << " recsel0.hasData "   << recsel0->hasData()
              ;


    unsigned int maxrec = getMaxRec();

    assert(sequence->hasItemShape(1,2));
    assert(phosel->hasItemShape(1,4));
    assert(recsel0->hasItemShape(maxrec,1,4));
    assert(sequence->getShape(0) == phosel->getShape(0));
    assert(sequence->getShape(0) == recsel0->getShape(0));

    Indexer<unsigned long long>* idx = new Indexer<unsigned long long>(sequence) ; 
    idx->indexSequence(OpticksConst::SEQHIS_NAME_, OpticksConst::SEQMAT_NAME_);

    assert(!phosel->hasData()) ; 

    phosel->zero();
    unsigned char* phosel_values = phosel->getValues() ;
    assert(phosel_values);
    idx->applyLookup<unsigned char>(phosel_values);



    NPY<unsigned char>* recsel1 = NPY<unsigned char>::make_repeat(phosel, maxrec ) ;
    recsel1->reshape(-1, maxrec, 1, 4);
    //recsel->save("/tmp/recsel.npy"); 

    // TODO: fix leak?, review recsel0 creation/zeroing/allocation
    setRecselData(recsel1);


    setHistoryIndex(idx->getHistoryIndex());
    setMaterialIndex(idx->getMaterialIndex());

    TIMER("indexPhotonsCPU");    
}




void OpticksEvent::saveIndex(bool verbose_)
{
    const char* udet = getUDet();

    NPYBase::setGlobalVerbose(verbose_);

    NPY<unsigned char>* ps = getPhoselData();
    NPY<unsigned char>* rs = getRecselData();

    assert(ps);
    assert(rs);

    ps->save("ps%s", m_typ,  m_tag, udet);
    rs->save("rs%s", m_typ,  m_tag, udet);

    NPYBase::setGlobalVerbose(false);

    std::string ixdir = getSpeciesDir("ix");
    LOG(info) << "OpticksEvent::saveIndex"
              << " ixdir " << ixdir
              << " seqhis " << m_seqhis
              << " seqmat " << m_seqmat
              << " bndidx " << m_bndidx
              ; 

    if(m_seqhis)
        m_seqhis->save(ixdir.c_str(), m_tag);        
    else
        LOG(warning) << "OpticksEvent::saveIndex no seqhis to save " ;

    if(m_seqmat)
        m_seqmat->save(ixdir.c_str(), m_tag);        
    else
        LOG(warning) << "OpticksEvent::saveIndex no seqmat to save " ;

    if(m_bndidx)
        m_bndidx->save(ixdir.c_str(), m_tag);        
    else
        LOG(warning) << "OpticksEvent::saveIndex no bndidx to save " ;
}

void OpticksEvent::loadIndex()
{
    std::string ixdir = getSpeciesDir("ix");
    m_seqhis = Index::load(ixdir.c_str(), m_tag, OpticksConst::SEQHIS_NAME_ );
    m_seqmat = Index::load(ixdir.c_str(), m_tag, OpticksConst::SEQMAT_NAME_ );  
    m_bndidx = Index::load(ixdir.c_str(), m_tag, OpticksConst::BNDIDX_NAME_ );
}


Index* OpticksEvent::loadNamedIndex( const char* typ, const char* tag, const char* udet, const char* name)
{
    const char* species = "ix" ; 
    std::string ixdir = speciesDir(species, udet, typ);
    Index* seqhis = Index::load(ixdir.c_str(), tag, name );
    return seqhis ; 
}

Index* OpticksEvent::loadHistoryIndex( const char* typ, const char* tag, const char* udet)
{
    return loadNamedIndex(typ, tag, udet, OpticksConst::SEQHIS_NAME_); 
}
Index* OpticksEvent::loadMaterialIndex( const char* typ, const char* tag, const char* udet)
{
    return loadNamedIndex(typ, tag, udet, OpticksConst::SEQMAT_NAME_); 
}
Index* OpticksEvent::loadBoundaryIndex( const char* typ, const char* tag, const char* udet)
{
    return loadNamedIndex(typ, tag, udet, OpticksConst::BNDIDX_NAME_); 
}








