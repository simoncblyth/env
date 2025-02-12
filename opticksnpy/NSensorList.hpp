#pragma once

#include <vector>
#include <set>
#include <string>
#include <map>

class NSensor ; 


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NSensorList {
    public:
        NSensorList();
        void load(const char* idpath, const char* ext="idmap");
        unsigned int getNumSensors();
        void dump(const char* msg="NSensorList::dump");
        std::string description();

    public:
        NSensor* getSensor(unsigned int index);
        NSensor* findSensorForNode(unsigned int nodeIndex);
    private:
        void read(const char* path);
        void add(NSensor* sensor);
        NSensor* createSensor(std::vector<std::string>& elem );
        unsigned int parseHexString(std::string& str);

    private:
        std::vector<NSensor*>    m_sensors ; 
        std::set<unsigned int>   m_ids ; 
        std::map<unsigned int, NSensor*>   m_nid2sen ; 

};

#include "NPY_TAIL.hh"

