#ifndef root2cjson_h
#define root2cjson_h

#include "TClass.h"
#include "TObject.h"
#ifdef __cplusplus
extern "C"
{
#endif

#include "cJSON.h"
cJSON* root2cjson( TClass* kls, TObject* obj );

#ifdef __cplusplus
}
#endif
#endif
