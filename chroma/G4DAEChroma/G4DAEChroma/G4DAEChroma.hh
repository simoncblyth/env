/*

Aiming for this class to present an unchanging 
interface, on top of the flexible internal machinery 
that can adopt different:

* transport mechanism
* hit collection 


*/
#ifndef G4DAECHROMA_H
#define G4DAECHROMA_H 1

#include <cstddef>


class G4DAEGeometry ;
class G4DAETransport ;
class G4DAESensDet ;
class G4Track ; 
class G4Run ;

class G4DAEChroma 
{
public:
    static G4DAEChroma* GetG4DAEChroma();
    static G4DAEChroma* GetG4DAEChromaIfExists();
protected:
    G4DAEChroma();
public:
    void Configure(const char* transport="G4DAECHROMA_CLIENT_CONFIG", const char* sensdet="DsPmtSensDet", const char* geometry=NULL);
    virtual ~G4DAEChroma();

    void SetGeometry(G4DAEGeometry* geo);
    G4DAEGeometry* GetGeometry();

    void SetTransport(G4DAETransport* tra);
    G4DAETransport* GetTransport();

    void SetSensDet(G4DAESensDet* sd);
    G4DAESensDet* GetSensDet();


    void BeginOfRun( const G4Run* run );
    void EndOfRun(   const G4Run* run );
    void Note( const char* msg );

    void ClearAll();
    void CollectPhoton(const G4Track* aPhoton );
    std::size_t Propagate(int batch_id);

 
private:
  // Singleton instance
  static G4DAEChroma* fG4DAEChroma;

private:
  // Photon Transport 
  G4DAETransport* fTransport ; 

  // Hit collection
  G4DAESensDet* fSensDet ; 

  // Geometry Transform cache, used to convert global to local coordinates
  G4DAEGeometry* fGeometry ; 


};


#endif


