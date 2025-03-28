#include "CFG4_BODY.hh"
// op --cgdmldetector
// op --ctestdetector

#include <cstdio>

// g4-
#include "G4PVPlacement.hh"

// npy-
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "NPY.hpp"
#include "NBoundingBox.hpp"

// okc-
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksQuery.hh"

// cfg4-
#include "CPropLib.hh"
#include "CTraverser.hh"
#include "CDetector.hh"

#include "PLOG.hh"



CDetector::CDetector(Opticks* opticks, OpticksQuery* query)
  : 
  m_opticks(opticks),
  m_query(query),
  m_resource(NULL),
  m_lib(NULL),
  m_top(NULL),
  m_traverser(NULL),
  m_bbox(NULL),
  m_verbosity(0),
  m_valid(true)
{
    init();
}


bool CDetector::isValid()
{
    return m_valid ; 
}

void CDetector::setValid(bool valid)
{
    m_valid = valid ; 
}





void CDetector::setTop(G4VPhysicalVolume* top)
{
    m_top = top ; 
    traverse(m_top);
}
G4VPhysicalVolume* CDetector::Construct()
{
    return m_top ; 
}
G4VPhysicalVolume* CDetector::getTop()
{
    return m_top ; 
}


CPropLib* CDetector::getPropLib()
{
    return m_lib ; 
}
NBoundingBox* CDetector::getBoundingBox()
{
    return m_bbox ; 
}

void CDetector::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}



void CDetector::init()
{
    LOG(trace) << "CDetector::init" ;

    m_resource = m_opticks->getResource();
    m_lib = new CPropLib(m_opticks);
    m_bbox = new NBoundingBox ;
}

void CDetector::traverse(G4VPhysicalVolume* /*top*/)
{
    // invoked from CGDMLDetector::init via setTop

    m_traverser = new CTraverser(m_top, m_bbox, m_query ); 
    m_traverser->Traverse();
    m_traverser->Summary("CDetector::traverse");
}

const glm::vec4& CDetector::getCenterExtent()
{
    return m_bbox->getCenterExtent() ; 
}

void CDetector::saveBuffers(const char* objname, unsigned int objindex)
{
    assert(m_traverser);

    std::string cachedir = m_opticks->getObjectPath(objname, objindex);

    NPY<float>* gtransforms = m_traverser->getGlobalTransforms(); 
    NPY<float>* ltransforms = m_traverser->getLocalTransforms(); 
    NPY<float>* center_extent = m_traverser->getCenterExtent(); 

    gtransforms->save(cachedir.c_str(), "gtransforms.npy");
    ltransforms->save(cachedir.c_str(), "ltransforms.npy");
    center_extent->save(cachedir.c_str(), "center_extent.npy");
}



unsigned int CDetector::getNumGlobalTransforms()
{
    assert(m_traverser);
    return m_traverser->getNumGlobalTransforms();
}
unsigned int CDetector::getNumLocalTransforms()
{
    assert(m_traverser);
    return m_traverser->getNumLocalTransforms();
}

glm::mat4 CDetector::getGlobalTransform(unsigned int index)
{
    assert(m_traverser);
    return m_traverser->getGlobalTransform(index);
}
glm::mat4 CDetector::getLocalTransform(unsigned int index)
{
    assert(m_traverser);
    return m_traverser->getLocalTransform(index);
}

NPY<float>* CDetector::getGlobalTransforms()
{
    assert(m_traverser);
    return m_traverser->getGlobalTransforms();
}

NPY<float>* CDetector::getLocalTransforms()
{
    assert(m_traverser);
    return m_traverser->getLocalTransforms();
}

const char* CDetector::getPVName(unsigned int index)
{
    assert(m_traverser);
    return m_traverser->getPVName(index);
}






void CDetector::dumpPV(const char* msg)
{
    LOG(info) << msg ; 

    typedef std::map<std::string, G4VPhysicalVolume*> MSV ; 

    for(MSV::const_iterator it=m_pvm.begin() ; it != m_pvm.end() ; it++)
    {
         std::string pvn = it->first ; 
         G4VPhysicalVolume* pv = it->second ;  

         std::cout << std::setw(40) << pvn 
                   << std::setw(40) << pv->GetName() 
                   << std::endl 
                   ;

    }
}

G4VPhysicalVolume* CDetector::getPV(const char* name)
{
    return m_pvm.count(name) == 1 ? m_pvm[name] : NULL ; 
}

CDetector::~CDetector()
{
    //printf("CDetector::~CDetector\n");
    //G4GeometryManager::GetInstance()->OpenGeometry();
    //printf("CDetector::~CDetector DONE\n");
}
