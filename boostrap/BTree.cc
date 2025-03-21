#include "BTree.hh"


#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "BJSONParser.hh"

#include <boost/filesystem.hpp>


#include "PLOG.hh"
// trace/debug/info/warning/error/fatal

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;



void BTree::saveTree(const pt::ptree& t , const char* path)
{
    fs::path fpath(path);
    std::string ext = fpath.extension().string();
    if(ext.compare(".json")==0)
        pt::write_json(path, t );
    else if(ext.compare(".ini")==0)
        pt::write_ini(path, t );
    else
        LOG(warning) << "saveTree cannot write to path with extension " << ext ; 
}


int BTree::loadTree(pt::ptree& t , const char* path)
{
    fs::path fpath(path);
    LOG(debug) << "BTree.loadTree: "
              << " load path: " << path;

    if (!(fs::exists(fpath ) && fs::is_regular_file(fpath))) {
        LOG(warning) << "BTree.loadTree: "
                     << "can't find file " << path;
        return 1;
    }
    std::string ext = fpath.extension().string();
    if(ext.compare(".json")==0)
        pt::read_json(path, t );
    else if(ext.compare(".ini")==0)
        pt::read_ini(path, t );
    else
        LOG(warning) << "readTree cannot read path with extension " << ext ; 

    return 0 ; 
}



