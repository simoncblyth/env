

#define OVECCOUNT 30    /* should be a multiple of 3 */
#define ENVVAR  "ENV_PRIVATE_PATH"
#define PATTERN "^local (?P<pname>.*)=(?P<pvalue>.*)"

int private_init();
int private_cleanup();
int private_dump();

const char* private_lookup( const char* key );


