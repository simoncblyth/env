#include <map>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <climits>

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NPart.hpp"
#include "GLMFormat.hpp"

#include "GVector.hh"
#include "GItemList.hh"
#include "GBndLib.hh"
#include "GParts.hh"

#include "PLOG.hh"


const char* GParts::CONTAINING_MATERIAL = "CONTAINING_MATERIAL" ;  
const char* GParts::SENSOR_SURFACE = "SENSOR_SURFACE" ;  

const char* GParts::SPHERE_ = "Sphere" ;
const char* GParts::TUBS_   = "Tubs" ;
const char* GParts::BOX_    = "Box" ;
const char* GParts::PRISM_  = "Prism" ;

const char* GParts::TypeName(unsigned int typecode)
{
    LOG(debug) << "GParts::TypeName " << typecode ; 
    switch(typecode)
    {
        case SPHERE:return SPHERE_ ; break ;
        case   TUBS:return TUBS_   ; break ;
        case    BOX:return BOX_    ; break ;
        case  PRISM:return PRISM_  ; break ;
        default:  assert(0) ; break ; 
    }
    return NULL ; 
}


GParts* GParts::combine(std::vector<GParts*> subs)
{
    GParts* parts = new GParts(); 
    GBndLib* bndlib = NULL ; 
    for(unsigned int i=0 ; i < subs.size() ; i++)
    {
        GParts* sp = subs[i];
        parts->add(sp);
        if(!bndlib) bndlib = sp->getBndLib(); 
    } 
    if(bndlib) parts->setBndLib(bndlib);
    return parts ; 
}


GParts* GParts::make(const npart& pt, const char* spec)
{
    NPY<float>* part = NPY<float>::make(1, NJ, NK );
    part->zero();

    part->setQuad( pt.q0.f, 0u, 0u );
    part->setQuad( pt.q1.f, 0u, 1u );
    part->setQuad( pt.q2.f, 0u, 2u );
    part->setQuad( pt.q3.f, 0u, 3u );

    GParts* gpt = new GParts(part, spec) ;
    char typecode = gpt->getTypeCode(0u);
    assert(typecode == BOX || typecode == SPHERE || typecode == PRISM);

    return gpt ; 
}

GParts* GParts::make(char typecode, glm::vec4& param, const char* spec)
{
    float size = param.w ;  
    gbbox bb(gfloat3(-size), gfloat3(size));  

    if(typecode == 'Z')
    {
        bb.min.z = param.x*param.w ; 
        bb.max.z = param.y*param.w ; 
    } 

    NPY<float>* part = NPY<float>::make(1, NJ, NK );
    part->zero();

    assert(BBMIN_K == 0 );
    assert(BBMAX_K == 0 );

    unsigned int i = 0u ; 
    part->setQuad( i, PARAM_J, param.x, param.y, param.z, param.w );
    part->setQuad( i, BBMIN_J, bb.min.x, bb.min.y, bb.min.z , 0.f );
    part->setQuad( i, BBMAX_J, bb.max.x, bb.max.y, bb.max.z , 0.f );

    GParts* pt = new GParts(part, spec) ;

    if( typecode == 'B' )     pt->setTypeCode(0u, BOX);
    else if(typecode == 'S')  pt->setTypeCode(0u, SPHERE);
    else if(typecode == 'Z')  pt->setTypeCode(0u, SPHERE);
    else if(typecode == 'M')  pt->setTypeCode(0u, PRISM);
    else
    {
        LOG(fatal) << "GParts::make bad typecode [" << typecode << "]" ; 
        assert(0) ; 
    }
    return pt ; 
} 



GParts::GParts(GBndLib* bndlib) 
      :
      m_part_buffer(NULL),
      m_bndspec(NULL),
      m_bndlib(bndlib),
      m_solid_buffer(NULL),
      m_closed(false),
      m_verbose(false)
{
      init() ; 
}

GParts::GParts(NPY<float>* buffer, const char* spec, GBndLib* bndlib) 
      :
      m_part_buffer(buffer),
      m_bndspec(NULL),
      m_bndlib(bndlib),
      m_solid_buffer(NULL),
      m_closed(false)
{
      init(spec) ; 
}
      
GParts::GParts(NPY<float>* buffer, GItemList* spec, GBndLib* bndlib) 
      :
      m_part_buffer(buffer),
      m_bndspec(spec),
      m_bndlib(bndlib),
      m_solid_buffer(NULL),
      m_closed(false)
{
      init() ; 
}

void GParts::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}

bool GParts::isClosed()
{
    return m_closed ; 
}



unsigned int GParts::getSolidNumParts(unsigned int solid_index)
{
    return m_parts_per_solid.count(solid_index)==1 ? m_parts_per_solid[solid_index] : 0 ; 
}


void GParts::setBndSpec(GItemList* bndspec)
{
    m_bndspec = bndspec ;
}
GItemList* GParts::getBndSpec()
{
    return m_bndspec ; 
}

void GParts::setBndLib(GBndLib* bndlib)
{
    m_bndlib = bndlib ; 
}
GBndLib* GParts::getBndLib()
{
    return m_bndlib ; 
}


void GParts::setSolidBuffer(NPY<unsigned int>* solid_buffer)
{
    m_solid_buffer = solid_buffer ; 
}
NPY<unsigned int>* GParts::getSolidBuffer()
{
    return m_solid_buffer ; 
}

void GParts::setPartBuffer(NPY<float>* part_buffer)
{
    m_part_buffer = part_buffer ; 
}
NPY<float>* GParts::getPartBuffer()
{
    return m_part_buffer ; 
}




void GParts::init(const char* spec)
{
    m_bndspec = new GItemList("GParts");
    m_bndspec->add(spec);
    init();
}
void GParts::init()
{
    if(m_part_buffer == NULL && m_bndspec == NULL)
    {
        LOG(trace) << "GParts::init creating empty part_buffer and bndspec " ; 

        NPY<float>* empty = NPY<float>::make(0, NJ, NK );
        empty->zero();

        m_part_buffer = empty ; 
        m_bndspec = new GItemList("GParts");
    } 

    LOG(debug) << "GParts::init" 
              << " part_buffer items " << m_part_buffer->getNumItems() 
              << " bndspec items " <<  m_bndspec->getNumItems() 
              ;

    assert(m_part_buffer->getNumItems() == m_bndspec->getNumItems() );
}

unsigned int GParts::getNumParts()
{
    if(!m_part_buffer)
    {
        LOG(error) << "GParts::getNumParts NULL part_buffer" ; 
        return 0 ; 
    }

    assert(m_part_buffer->getNumItems() == m_bndspec->getNumItems() );
    return m_part_buffer->getNumItems() ;
}

void GParts::add(GParts* other)
{
    unsigned int n0 = getNumParts();
    //unsigned int nextPartIndex = n0 > 0 ? getIndex(n0 - 1) : 0 ;
    //assert(lastPartIndex == n0 - 1 ); 

    m_bndspec->add(other->getBndSpec());
    m_part_buffer->add(other->getPartBuffer());

    unsigned int n1 = getNumParts();
    for(unsigned int p=n0 ; p < n1 ; p++)
    {
        setIndex(p, p);
    }

    LOG(debug) << "GParts::add"
              << " n0 " << n0  
              << " n1 " << n1
              ;  
}



void GParts::setContainingMaterial(const char* material)
{
    // for flexibility persisted GParts should leave the outer containing material
    // set to a default marker name, to allow the GParts to be placed within other geometry

    m_bndspec->replaceField(0, GParts::CONTAINING_MATERIAL, material );
}

void GParts::setSensorSurface(const char* surface)
{
    m_bndspec->replaceField(1, GParts::SENSOR_SURFACE, surface ) ; 
    m_bndspec->replaceField(2, GParts::SENSOR_SURFACE, surface ) ; 
}



void GParts::close()
{
    registerBoundaries();
    makeSolidBuffer(); 
}

void GParts::registerBoundaries()
{
   assert(m_bndlib); 
   unsigned int nbnd = m_bndspec->getNumKeys() ; 
   assert( getNumParts() == nbnd );
   for(unsigned int i=0 ; i < nbnd ; i++)
   {
       const char* spec = m_bndspec->getKey(i);
       unsigned int boundary = m_bndlib->addBoundary(spec);
       setBoundary(i, boundary);

       if(m_verbose)
       LOG(info) << "GParts::registerBoundaries " 
                << std::setw(3) << i 
                << " " << std::setw(30) << spec
                << " --> "
                << std::setw(4) << boundary 
                << " " << std::setw(30) << m_bndlib->shortname(boundary)
                ;

   } 
}

void GParts::makeSolidBuffer()
{
    m_parts_per_solid.clear();
    unsigned int nmin(INT_MAX) ; 
    unsigned int nmax(0) ; 

    // count parts for each nodeindex
    for(unsigned int i=0; i < getNumParts() ; i++)
    {
        unsigned int nodeIndex = getNodeIndex(i);

        LOG(debug) << "GParts::solidify"
                   << " i " << std::setw(3) << i  
                   << " nodeIndex " << std::setw(3) << nodeIndex
                   ;  
                     
        m_parts_per_solid[nodeIndex] += 1 ; 

        if(nodeIndex < nmin) nmin = nodeIndex ; 
        if(nodeIndex > nmax) nmax = nodeIndex ; 
    }

    unsigned int num_solids = m_parts_per_solid.size() ;
    //assert(nmax - nmin == num_solids - 1);  // expect contiguous node indices
    if(nmax - nmin != num_solids - 1)
    {
        LOG(warning) << "GParts::solidify non-contiguous node indices"
                     << " nmin " << nmin 
                     << " nmax " << nmax
                     << " num_solids " << num_solids 
                     ; 
    }


    guint4* solidinfo = new guint4[num_solids] ;

    typedef std::map<unsigned int, unsigned int> UU ; 
    unsigned int part_offset = 0 ; 
    unsigned int n = 0 ; 
    for(UU::const_iterator it=m_parts_per_solid.begin() ; it != m_parts_per_solid.end() ; it++)
    {
        unsigned int node_index = it->first ; 
        unsigned int parts_for_solid = it->second ; 

        guint4& si = *(solidinfo+n) ;

        si.x = part_offset ; 
        si.y = parts_for_solid ;
        si.z = node_index ; 
        si.w = 0 ;              

        LOG(debug) << "GParts::solidify solidinfo " << si.description() ;       

        part_offset += parts_for_solid ; 
        n++ ; 
    }

    NPY<unsigned int>* buf = NPY<unsigned int>::make( num_solids, 4 );
    buf->setData((unsigned int*)solidinfo);
    delete [] solidinfo ; 

    setSolidBuffer(buf);
}


void GParts::dumpSolidInfo(const char* msg)
{
    LOG(info) << msg << " (part_offset, parts_for_solid, solid_index, 0) " ;
    for(unsigned int i=0 ; i < getNumSolids(); i++)
    {
        guint4 si = getSolidInfo(i);
        LOG(info) << si.description() ;
    }
}


void GParts::Summary(const char* msg)
{
    LOG(info) << msg 
              << " num_parts " << getNumParts() 
              << " num_solids " << getNumSolids()
              ;
 
    typedef std::map<unsigned int, unsigned int> UU ; 
    for(UU::const_iterator it=m_parts_per_solid.begin() ; it!=m_parts_per_solid.end() ; it++)
    {
        unsigned int solid_index = it->first ; 
        unsigned int nparts = it->second ; 
        unsigned int nparts2 = getSolidNumParts(solid_index) ; 
        printf("%2u : %2u \n", solid_index, nparts );
        assert( nparts == nparts2 );
    }

    for(unsigned int i=0 ; i < getNumParts() ; i++)
    {
        std::string bn = getBoundaryName(i);
        printf(" part %2u : node %2u type %2u boundary [%3u] %s  \n", i, getNodeIndex(i), getTypeCode(i), getBoundary(i), bn.c_str() ); 
    }
}




unsigned int GParts::getNumSolids()
{
    return m_solid_buffer ? m_solid_buffer->getShape(0) : 0 ; 
}
const char* GParts::getTypeName(unsigned int part_index)
{
    unsigned int code = getTypeCode(part_index);
    return GParts::TypeName(code);
}
     
float* GParts::getValues(unsigned int i, unsigned int j, unsigned int k)
{
    float* data = m_part_buffer->getValues();
    float* ptr = data + i*NJ*NK+j*NJ+k ;
    return ptr ; 
}
     
gfloat3 GParts::getGfloat3(unsigned int i, unsigned int j, unsigned int k)
{
    float* ptr = getValues(i,j,k);
    return gfloat3( *ptr, *(ptr+1), *(ptr+2) ); 
}
guint4 GParts::getSolidInfo(unsigned int isolid)
{
    unsigned int* data = m_solid_buffer->getValues();
    unsigned int* ptr = data + isolid*SK  ;
    return guint4( *ptr, *(ptr+1), *(ptr+2), *(ptr+3) );
}
gbbox GParts::getBBox(unsigned int i)
{
   gfloat3 min = getGfloat3(i, BBMIN_J, BBMIN_K );  
   gfloat3 max = getGfloat3(i, BBMAX_J, BBMAX_K );  
   gbbox bb(min, max) ; 
   return bb ; 
}

void GParts::enlargeBBoxAll(float epsilon)
{
   for(unsigned int part=0 ; part < getNumParts() ; part++) enlargeBBox(part, epsilon);
}

void GParts::enlargeBBox(unsigned int part, float epsilon)
{
    float* pmin = getValues(part,BBMIN_J,BBMIN_K);
    float* pmax = getValues(part,BBMAX_J,BBMAX_K);

    glm::vec3 min = glm::make_vec3(pmin) - glm::vec3(epsilon);
    glm::vec3 max = glm::make_vec3(pmax) + glm::vec3(epsilon);
 
    *(pmin+0 ) = min.x ; 
    *(pmin+1 ) = min.y ; 
    *(pmin+2 ) = min.z ; 

    *(pmax+0 ) = max.x ; 
    *(pmax+1 ) = max.y ; 
    *(pmax+2 ) = max.z ; 

    LOG(debug) << "GParts::enlargeBBox"
              << " part " << part 
              << " epsilon " << epsilon
              << " min " << gformat(min) 
              << " max " << gformat(max)
              ; 

}


unsigned int GParts::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < getNumParts() );
    unsigned int l=0u ; 
    return m_part_buffer->getUInt(i,j,k,l);
}
void GParts::setUInt(unsigned int i, unsigned int j, unsigned int k, unsigned int value)
{
    assert(i < getNumParts() );
    unsigned int l=0u ; 
    m_part_buffer->setUInt(i,j,k,l, value);
}



unsigned int GParts::getNodeIndex(unsigned int part)
{
    return getUInt(part, NODEINDEX_J, NODEINDEX_K);
}
unsigned int GParts::getTypeCode(unsigned int part)
{
    return getUInt(part, TYPECODE_J, TYPECODE_K);
}
unsigned int GParts::getIndex(unsigned int part)
{
    return getUInt(part, INDEX_J, INDEX_K);
}
unsigned int GParts::getBoundary(unsigned int part)
{
    return getUInt(part, BOUNDARY_J, BOUNDARY_K);
}
unsigned int GParts::getFlags(unsigned int part)
{
    return getUInt(part, FLAGS_J, FLAGS_K);
}



void GParts::setNodeIndex(unsigned int part, unsigned int nodeindex)
{
    setUInt(part, NODEINDEX_J, NODEINDEX_K, nodeindex);
}
void GParts::setTypeCode(unsigned int part, unsigned int typecode)
{
    setUInt(part, TYPECODE_J, TYPECODE_K, typecode);
}
void GParts::setIndex(unsigned int part, unsigned int index)
{
    setUInt(part, INDEX_J, INDEX_K, index);
}
void GParts::setBoundary(unsigned int part, unsigned int boundary)
{
    setUInt(part, BOUNDARY_J, BOUNDARY_K, boundary);
}
void GParts::setFlags(unsigned int part, unsigned int flags)
{
    setUInt(part, FLAGS_J, FLAGS_K, flags);
}


void GParts::setBoundaryAll(unsigned int boundary)
{
    for(unsigned int i=0 ; i < getNumParts() ; i++) setBoundary(i, boundary);
}




std::string GParts::getBoundaryName(unsigned int part)
{
    unsigned int boundary = getBoundary(part);
    std::string name = m_bndlib ? m_bndlib->shortname(boundary) : "" ;
    return name ;
}




void GParts::dump(const char* msg)
{
    dumpSolidInfo(msg);

    NPY<float>* buf = m_part_buffer ; 
    assert(buf);
    assert(buf->getDimensions() == 3);

    unsigned int ni = buf->getShape(0) ;
    unsigned int nj = buf->getShape(1) ;
    unsigned int nk = buf->getShape(2) ;

    assert( nj == NJ );
    assert( nk == NK );

    float* data = buf->getValues();

    uif_t uif ; 

    for(unsigned int i=0; i < ni; i++)
    {   
       unsigned int tc = getTypeCode(i);
       unsigned int id = getIndex(i);
       unsigned int bnd = getBoundary(i);
       std::string  bn = getBoundaryName(i);
       unsigned int flg = getFlags(i);
       const char*  tn = getTypeName(i);

       for(unsigned int j=0 ; j < NJ ; j++)
       {   
          for(unsigned int k=0 ; k < NK ; k++) 
          {   
              uif.f = data[i*NJ*NK+j*NJ+k] ;
              if( j == TYPECODE_J && k == TYPECODE_K )
              {
                  assert( uif.u == tc );
                  printf(" %10u (%s) ", uif.u, tn );
              } 
              else if( j == INDEX_J && k == INDEX_K)
              {
                  assert( uif.u == id );
                  printf(" %6u id   " , uif.u );
              }
              else if( j == BOUNDARY_J && k == BOUNDARY_K)
              {
                  assert( uif.u == bnd );
                  printf(" %6u bnd  ", uif.u );
              }
              else if( j == FLAGS_J && k == FLAGS_K)
              {
                  assert( uif.u == flg );
                  printf(" %6u flg  ", uif.u );
              }
              else if( j == NODEINDEX_J && k == NODEINDEX_K)
                  printf(" %10d (nodeIndex) ", uif.i );
              else
                  printf(" %10.4f ", uif.f );
          }   

          if( j == BOUNDARY_J ) printf(" bn %s ", bn.c_str() );
          printf("\n");
       }   
       printf("\n");
    }   

}


