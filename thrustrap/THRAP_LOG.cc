
#include <plog/Log.h>

#include "THRAP_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void THRAP_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void THRAP_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

