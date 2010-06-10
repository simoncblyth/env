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
//
// returns pointers to chars stored in glib hashtable that is created by the init and deleted by the cleanup
// so dont cleanup until you no longer need to use the chars 
const char* private_lookup( const char* key );
const char* private_lookup_default( const char* key , const char* def  );

// miscellaneous functions that do not need the init or cleanup
//  tfmt is strftime format and afmt combines the hostname and time strings 
int private_getftime( char* buffer , size_t max ,  const char* tfmt );
int private_getuserhostftime( char* buffer , size_t max , const char* tfmt , const char* afmt );

char* private_hostname();
char* private_username();
char* private_userhost();


#ifdef __cplusplus
}
#endif
#endif

