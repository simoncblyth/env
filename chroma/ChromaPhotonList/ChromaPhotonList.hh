////////////////////////////////////////////////////////////////////
/// \class ChromaPhotonList
/// \brief List of photons to send to a Chroma server for propagation
///
/// \author A. Mastbaum <mastbaum@hep.upenn.edu>
///
/// REVISION HISTORY:
///
/// \detail Separate arrays are space-efficient, and it's TObject
///         for purposes of serialization 
///
////////////////////////////////////////////////////////////////////

#ifndef __ChromaPhotonList__
#define __ChromaPhotonList__

#include <TObject.h>
#include <vector>

#ifdef WITH_GEANT4
#include <G4ThreeVector.hh>
#endif


class ChromaPhotonList : public TObject {

public:
  ChromaPhotonList();
  virtual ~ChromaPhotonList();
  void Print(Option_t* option = "") const ; 

#ifdef WITH_GEANT4
  inline void AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid=-1); 
  void GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const;
#endif

  void Details() const ;
  void GetPhoton(size_t index, 
                 float& _x,    float& _y, float& _z, 
                 float& _px, float& _py, float& _pz, 
                 float& _polx, float& _poly ,float& _polz, 
                 float& _t, float& _wavelength, int& _pmtid ) const ;

  void ClearAll();
  void AddPhoton(float _x, float _y, float _z,  
                        float _px, float _py, float _pz, 
                        float _polx, float _poly, float _polz, float _t, 
                        float _wavelength, int _pmtid=-1);

  void FromArrays(float* __x,    float* __y,    float* __z,
                  float* __px,   float* __py,   float* __pz,
                  float* __polx, float* __poly, float* __polz,
                  float* __t, float* __wavelength, int* __pmtid, int nphotons);


  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> px;
  std::vector<float> py;
  std::vector<float> pz;
  std::vector<float> polx;
  std::vector<float> poly;
  std::vector<float> polz;
  std::vector<float> t;
  std::vector<float> wavelength;
  std::vector<int> pmtid;


  ClassDef(ChromaPhotonList, 1);
};

#endif

