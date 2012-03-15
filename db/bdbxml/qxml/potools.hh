
#ifndef POTOOLS_HH
#define POTOOLS_HH

#include <map>
#include <string>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;
typedef vector<string> svec ;
typedef map<string, string> ssmap;
typedef map<string, ssmap> sssmap;

void kv_zip( po::variables_map& vm , ssmap& ss , string kname , string vname );
void kv_pluck( po::variables_map& vm , ssmap& ss , int ndot );
void kv_dump( string msg, ssmap& ss );
void cfg_dump( sssmap& m );
void vm_dump( po::variables_map& vm );

#endif
