#ifndef private_h
#define private_h

#ifdef __cplusplus
extern "C"
{
#endif

#define OVECCOUNT 30    /* should be a multiple of 3 */
#define ENVVAR  "ENV_PRIVATE_PATH"
#define PATTERN "^local (?P<pname>.*)=(?P<pvalue>.*)"

int private_init();
int private_cleanup();
int private_dump();

// implemented in C for widest applicability, including from notifmq 

const char* private_lookup( const char* key );
const char* private_lookup_default( const char* key , const char* def  );

#ifdef __cplusplus
}
#endif
#endif

