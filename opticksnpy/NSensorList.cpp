#include <cassert>
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>


#include "NSensorList.hpp"
#include "NSensor.hpp"

#include "PLOG.hh"

namespace fs = boost::filesystem;


/*

simon:DayaBay_VGDX_20140414-1300 blyth$ head -10 g4_00.idmap
# GiGaRunActionExport::WriteIdMap fields: index,pmtid,pmtid(hex),pvname  npv:12230
0 0 0 (0,0,0) (1,0,0)(0,1,0)(0,0,1) Universe
1 0 0 (664494,-449556,2110) (-0.543174,-0.83962,0)(0.83962,-0.543174,0)(0,0,1) /dd/Structure/Sites/db-rock
2 0 0 (661994,-449056,-5390) (-0.543174,-0.83962,0)(0.83962,-0.543174,0)(0,0,1) /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop
3 0 0 (664494,-449556,2088) (-0.543174,-0.83962,0)(0.83962,-0.543174,0)(0,0,1) /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover
4 0 0 (668975,-437058,-683.904) (-0.53472,-0.84503,0)(0.84503,-0.53472,0)(0,0,1) /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1
5 0 0 (668985,-437063,-683.904) (-0.53472,-0.84503,0)(0.84503,-0.53472,0)(0,0,1) /dd/Geometry/RPC/lvRPCMod#pvRPCFoam
6 0 0 (668985,-437063,-669.904) (-0.53472,-0.84503,0)(0.84503,-0.53472,0)(0,0,1) /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit
7 0 0 (668985,-437063,-669.904) (-0.53472,-0.84503,0)(0.84503,-0.53472,0)(0,0,1) /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14
8 0 0 (-437063,-669895,-669.904) (-0.84503,0.53472,0)(-0.53472,-0.84503,0)(0,0,1) /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit
simon:DayaBay_VGDX_20140414-1300 blyth$ 
simon:DayaBay_VGDX_20140414-1300 blyth$ grep PMT g4_00.idmap | head -10
3200 16843009 1010101 (8842.5,532069,599609) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
3201 16843009 1010101 (8842.5,532069,599609) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
3202 16843009 1010101 (8842.5,532069,599540) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
3203 16843009 1010101 (8842.5,532069,599690) (3.96846e-17,0.761538,-0.64812)(-4.66292e-17,0.64812,0.761538)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
3206 16843010 1010102 (8842.5,668528,441547) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
3207 16843010 1010102 (8842.5,668528,441547) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
3208 16843010 1010102 (8842.5,668528,441478) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
3209 16843010 1010102 (8842.5,668528,441628) (5.04009e-17,0.567844,-0.823136)(-3.47693e-17,0.823136,0.567844)(1,0,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode
3212 16843011 1010103 (8842.5,759428,253553) (5.76825e-17,0.335452,-0.942057)(-2.05398e-17,0.942057,0.335452)(1,6.16298e-33,6.12303e-17) /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
3213 16843011 1010103 (8842.5,759428,253553) (5.76825e-17,0.335452,-0.942057)(-2.05398e-17,0.942057,0.335452)(1,6.16298e-33,6.12303e-17) /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
simon:DayaBay_VGDX_20140414-1300 blyth$ 

*/




NSensorList::NSensorList()
{
}
NSensor* NSensorList::getSensor(unsigned int index)
{
    return index < m_sensors.size() ? m_sensors[index] : NULL ; 
}
unsigned int NSensorList::getNumSensors()
{
    return m_sensors.size();
}



void NSensorList::load(const char* idpath_, const char* ext)
{
    if(!idpath_) return ;

    fs::path idpath(idpath_);
    std::string name = idpath.filename().string() ;
    fs::path pdir = idpath.parent_path();

    std::vector<std::string> elem ;
    boost::split(elem, name, boost::is_any_of("."));

    if(elem.size() != 3 )
    {
        LOG(error) << "NSensorList::load"
                   << " idpath is expected to be in 3-parts separted by dot eg  g4_00.gdasdyig3736781.dae "
                   << " idpath " << idpath_
                    ; 

        return ; 
    }
    

    assert(elem.size() == 3 );
    elem.erase(elem.begin() + 1); // remove hex digest

    std::string daename = boost::algorithm::join(elem, ".");

    fs::path daepath(pdir);
    daepath /= daename ;

    elem[1] = ext ; 
    std::string idmname = boost::algorithm::join(elem, ".");

    fs::path idmpath(pdir);
    idmpath /= idmname ; 

    LOG(debug) << "NSensorList::load "
              << "\n idpath:   " << idpath.string() 
              << "\n pdir:     " << pdir.string() 
              << "\n filename: " << name 
              << "\n daepath:  " << daepath.string() 
              << "\n idmpath:  " << idmpath.string() 
              ;   

    read(idmpath.string().c_str());
}


void NSensorList::read(const char* path)
{
    std::ifstream in(path, std::ios::in);
    if(!in.is_open()) 
    {   
        LOG(fatal) << "NSensorList::read failed to open " << path ; 
        return ;
    }   

    typedef boost::tokenizer< boost::char_separator<char> > Tok_t;
    boost::char_separator<char> delim(" ");

    std::string line ; 
    std::vector<std::string> elem ; 

    unsigned int count(0);
    while(std::getline(in, line))
    {   
        if(line[0] == '#') continue ; 

        Tok_t tok(line, delim) ;
        elem.assign(tok.begin(), tok.end());
        NSensor* sensor = createSensor(elem);
        if(sensor) add(sensor);

        //if(count < 10) printf("[%lu] %s \n", elem.size(), line.c_str());
        count++;
    }
    in.close();
    LOG(debug) << "NSensorList::read " 
              << " path " << path 
              << " desc " << description() 
              ; 
}


std::string NSensorList::description()
{
    std::stringstream ss ; 
    ss << "NSensorList: " 
       << " NSensor count " << m_sensors.size()
       << " distinct identier count " << m_ids.size() 
      ;
    return ss.str();
}


NSensor* NSensorList::createSensor(std::vector<std::string>& elem )
{
    assert(elem.size() == 6 );

    unsigned int nodeIndex  = boost::lexical_cast<unsigned int>(elem[0]); 
    unsigned int id         = boost::lexical_cast<unsigned int>(elem[1]); 
    unsigned int id_hex     = parseHexString(elem[2]);
    assert(id == id_hex );

    NSensor* sensor(NULL);
    if( id > 0 )
    {
        unsigned int index = m_sensors.size() ;    // 0-based in tree node order
        const char* nodeName = elem[5].c_str() ; 
        sensor = new NSensor(index, id, nodeName, nodeIndex);
    }
    return sensor ; 
}


void NSensorList::add(NSensor* sensor)
{
    assert(sensor);

    m_sensors.push_back(sensor);  

    m_ids.insert(sensor->getId());

    unsigned int nid = sensor->getNodeIndex(); 

    assert(m_nid2sen.count(nid) == 0 && "there should only ever be one NSensor for each node index");

    m_nid2sen[nid] = sensor ; 
}


unsigned int NSensorList::parseHexString(std::string& str)
{
    unsigned int x;   
    std::stringstream ss;
    ss << std::hex << str ;
    ss >> x;
    return x ; 
}


void NSensorList::dump(const char* msg)
{
    unsigned int nsen = getNumSensors();
    printf("%s : %u sensors \n", msg, nsen);
    for(unsigned int i=0 ; i < nsen ; i++)
    {
        NSensor* sensor = getSensor(i);
        assert(sensor);
        printf("%s\n", sensor->description().c_str()) ;

        NSensor* check = findSensorForNode(sensor->getNodeIndex());
        assert(check == sensor);
    }
}

NSensor* NSensorList::findSensorForNode(unsigned int nodeIndex)
{
    return m_nid2sen.count(nodeIndex) == 1 ? m_nid2sen[nodeIndex] : NULL ; 
}



