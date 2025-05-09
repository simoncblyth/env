
#include <plog/Log.h>

// Helper macro to mark functions exported from the library.
#if defined _MSC_VER || defined __CYGWIN__
#   define EXPORT __declspec(dllexport)
#else
#   define EXPORT __attribute__ ((visibility ("default")))
#endif

// Function that initializes the logger in the shared library. 
extern "C" void EXPORT foo_initialize(plog::Severity severity, plog::IAppender* appender)
{
    plog::init(severity, appender); // Initialize the shared library logger.
}

// Function that produces a log message.
extern "C" void EXPORT foo_check()
{
    LOGI << "Hello from shared lib!";
}
