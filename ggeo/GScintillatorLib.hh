#pragma once

#include <vector>
class Opticks ; 
class GItemList ;
template <typename T> class GPropertyMap ;

#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GScintillatorLib : public GPropertyLib {
    public:
        static const char* slow_component; 
        static const char* fast_component; 
        static const char* keyspec ;
    public:
        void save();
        void dump(const char* msg="GScintillatorLib::dump");
        static GScintillatorLib* load(Opticks* cache);
    public:
        GScintillatorLib(Opticks* cache, unsigned int icdf_length=4096);
        void Summary(const char* msg="GScintillatorLib::Summary");
    public:
        void add(GPropertyMap<float>* scint);
        unsigned int getNumScintillators();
    public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
       void import();
       void sort();
       NPY<float>* createBuffer();
       GItemList*  createNames();
    private:
        void init();
    public:
        GProperty<float>* constructReemissionCDF(GPropertyMap<float>* pmap);
        GProperty<float>* constructInvertedReemissionCDF(GPropertyMap<float>* pmap);
    private:
        unsigned int m_icdf_length ; 

};

#include "GGEO_TAIL.hh"


