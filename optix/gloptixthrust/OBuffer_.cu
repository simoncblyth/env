
#include <optix_world.h>

#include <iostream>
#include <iterator>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "OBuffer.hh"
#include "CResource.hh"

#include <cuda_runtime.h>


const char* OBuffer::UNMAPPED_        = "UNMAPPED" ;
const char* OBuffer::GLToCUDA_        = "GLToCUDA        (getMappedPointer) " ;
const char* OBuffer::GLToCUDAToOptiX_ = "GLToCUDAToOptiX (getMappedPointer,createBufferForCUDA,setDevicePointer) " ;
const char* OBuffer::GLToOptiX_       = "GLToOptiX       (createBufferFromGLBO)";
const char* OBuffer::OptiXToCUDA_     = "OptiXToCUDA     (getDevicePointer) " ;
const char* OBuffer::OptiX_           = "OptiX           (createBuffer) " ;

void OBuffer::init()
{
    printf("OBuffer::init %u %s \n", m_buffer_id, m_buffer_name );
    m_resource = m_buffer_id == 0 ? NULL : new CResource(m_buffer_id, m_access  );
}

void OBuffer::streamSync()
{
    assert(m_resource);
    m_resource->streamSync();
}

CBufSpec OBuffer::map(OBuffer::Mapping_t mapping)
{
    m_mapping = mapping ; 
    //printf("OBuffer::map %s %d\n", getMappingDescription(), m_buffer_id);
    preqs();
    switch(m_mapping)
    {
        case        UNMAPPED: assert(0)           ;break;
        case        GLToCUDA: mapGLToCUDA()       ;break;
        case GLToCUDAToOptiX: mapGLToCUDAToOptiX();break;
        case       GLToOptiX: mapGLToOptiX()      ;break;
        case     OptiXToCUDA: mapOptiXToCUDA()    ;break;
        case           OptiX: mapOptiX()          ;break;
    }
    return m_bufspec ; 
}

void OBuffer::unmap()
{
    //printf("OBuffer::unmap %s %d\n", getMappingDescription(), m_buffer_id);
    preqs();
    switch(m_mapping)
    {
        case        UNMAPPED: assert(0)              ;break;
        case        GLToCUDA: unmapGLToCUDA()        ;break;
        case GLToCUDAToOptiX: unmapGLToCUDAToOptiX() ;break;
        case       GLToOptiX: unmapGLToOptiX()       ;break;
        case     OptiXToCUDA: unmapOptiXToCUDA()     ;break;
        case           OptiX: unmapOptiX()           ;break;
    }
    m_mapping = UNMAPPED ; 
}

const char* OBuffer::getMappingDescription()
{
    const char* desc = NULL ; 
    switch(m_mapping)
    {
        case        UNMAPPED: desc = UNMAPPED_        ;break;
        case        GLToCUDA: desc = GLToCUDA_        ;break;
        case GLToCUDAToOptiX: desc = GLToCUDAToOptiX_ ;break;
        case       GLToOptiX: desc = GLToOptiX_       ;break;
        case     OptiXToCUDA: desc = OptiXToCUDA_     ;break;
        case           OptiX: desc = OptiX_           ;break;
    }
    return desc ; 
}

void OBuffer::preqs()
{
    if(m_mapping == GLToCUDA || m_mapping == GLToCUDAToOptiX)
    {
        assert(m_buffer_id > 0 );
        assert(m_resource);
    } 
    else if(m_mapping == GLToOptiX )
    {
        assert(m_buffer_id > 0 );
    }
    else if(m_mapping == OptiX )
    {
        assert(m_buffer_id == 0 );
    }
}


void OBuffer::mapGLToCUDA()
{
    m_bufspec = m_resource->mapGLToCUDA();
    m_bufspec.size    = m_size ; 
}
void OBuffer::unmapGLToCUDA()
{
    m_resource->unmapGLToCUDA();
}

void OBuffer::mapGLToCUDAToOptiX()
{
    m_bufspec = m_resource->mapGLToCUDA();

    CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(m_bufspec.dev_ptr) ; 
    m_buffer = m_context->createBufferForCUDA(m_type, m_format, m_size);
    m_buffer->setDevicePointer(m_device, cu_ptr );

    //fillBufSpec( dptr );
    //assert( m_bufspec.num_bytes == m_resource->getNumBytes()); 
    m_bufspec.size = m_size ; 

    m_context[m_buffer_name]->setBuffer(m_buffer);
}

void OBuffer::unmapGLToCUDAToOptiX()
{
    m_buffer->markDirty();  // before the unmap : think of the unmap as potentially copying back to GL
    unmapGLToCUDA();
}

void OBuffer::mapOptiX()
{
    m_buffer = m_context->createBuffer(m_type, m_format, m_size);
    fillBufSpec(NULL);
    m_context[m_buffer_name]->setBuffer(m_buffer);
}
void OBuffer::unmapOptiX()
{
}



void OBuffer::mapGLToOptiX()
{
    m_buffer = m_context->createBufferFromGLBO(m_type, m_buffer_id);
    m_buffer->setFormat( m_format );
    m_buffer->setSize( m_size );

    fillBufSpec( NULL );

    m_context[m_buffer_name]->setBuffer(m_buffer);
}
void OBuffer::unmapGLToOptiX()
{
}

void OBuffer::mapOptiXToCUDA()
{
    if(!m_buffer->get())
    {
         printf("OBuffer::mapOptiXToCUDA FAILED : no OptiX buffer\n");
         return ; 
    }

    CUdeviceptr dev_ptr;
    m_buffer->getDevicePointer(m_device, &dev_ptr);

    fillBufSpec( (void*)dev_ptr );

    //m_bufspec.Summary("OBuffer::mapOptiXToCUDA (getDevicePointer) bufspec");
}

void OBuffer::unmapOptiXToCUDA()
{
    //printf("OBuffer::unmapOptiXToCUDA (markDirty) \n");  // when is this acted upon ? next launch perhaps ? need dummy launch maybe
    m_buffer->markDirty(); 
}





unsigned int OBuffer::getBufferSize()
{
    RTsize width, height, depth ; 
    m_buffer->getSize(width, height, depth);
    unsigned int size = width*height*depth ; 
    assert(size == m_size );
    return size ; 
}
unsigned int OBuffer::getElementSize()
{
    size_t element_size ; 
    rtuGetSizeForRTformat( m_buffer->getFormat(), &element_size); // hmm this is in bytes
    return element_size ; 
}
void OBuffer::fillBufSpec(void* dev_ptr)
{
    unsigned int element_size = getElementSize();
    unsigned int size = getBufferSize();

    m_bufspec.dev_ptr = dev_ptr ; 
    m_bufspec.size = size ; 
    m_bufspec.num_bytes = element_size*size  ;
}


