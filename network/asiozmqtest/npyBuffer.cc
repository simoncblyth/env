#include "npyBuffer.hh"
#include "stdlib.h"
#include "string.h"

npyBuffer::npyBuffer( char* bytes, size_t size) 
    :
    m_bytes(NULL), 
    m_size(size)
{
    m_bytes = new char[m_size];
    memcpy((void*)m_bytes, bytes, size ); 
}

npyBuffer::~npyBuffer()
{
    delete m_bytes ;
} 



