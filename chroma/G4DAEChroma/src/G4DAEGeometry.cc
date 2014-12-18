#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"

#include "G4AffineTransform.hh"
#include "G4TransportationManager.hh"
#include "G4NavigationHistory.hh"
#include "G4TouchableHistory.hh"
#include "G4Material.hh"

#ifdef EXPORT_G4GDML
#include "G4GDMLParser.hh"
#endif

#include <stdlib.h>    
#include <iostream>    
#include <string>    
#include <sstream>    
#include <iomanip>    
#include <vector>

using namespace std ; 


G4DAEGeometry::G4DAEGeometry() :  m_pvcount(0), m_sdcount(0)
{ 
}

G4DAEGeometry::~G4DAEGeometry()
{
}


G4DAEGeometry* G4DAEGeometry::MakeGeometry( const char* geometry )
{
   return ( strcmp(geometry,"MEMORY") == 0 ) ? Load(NULL) : LoadFromGDML(geometry) ;
}



#ifdef EXPORT_G4GDML
G4DAEGeometry* G4DAEGeometry::LoadFromGDML( const char* geokey, G4VSensitiveDetector* sd )
{
   const char* geopath = getenv(geokey);
   if(geopath == NULL )
   {
      printf("G4DAEGeometry::LoadFromGDML geokey %s : missing : use \"export-;export-export\" to define  \n", geokey );
      return NULL;
   }   
   printf("geokey %s geopath %s \n", geokey, geopath ); 

   string archivedir(geopath);
   archivedir += ".cache" ; 

   G4DAEGeometry* geo = new G4DAEGeometry();

   if( G4DAETransformCache::Exists(archivedir.c_str()))
   {
       G4DAETransformCache* cache = G4DAETransformCache::Load( archivedir.c_str() ); 
   } 
   else
   {
       G4GDMLParser fParser ; 
       fParser.Read(geopath,false);

       G4VPhysicalVolume* top = fParser.GetWorldVolume();       
       G4LogicalVolume* ltop = top->GetLogicalVolume();

       if( sd != NULL )
       {  
           string fakesd(geokey);
           fakesd += "_FAKESD" ;
   
           geo->AddSensitiveLVNames(fakesd.c_str(),';');
           geo->DumpSensitiveLVNames();

           PVStack_t pvStack ;  
           geo->FakeAssignSensitive( ltop, pvStack, sd );
           // TODO: check fake SD assignments is accurate mockup 
       } 
       G4DAETransformCache* cache = geo->CreateTransformCache( top ); 
       cache->Archive( archivedir.c_str() ); 
   } 
   return geo;
}
#else
G4DAEGeometry* G4DAEGeometry::LoadFromGDML( const char*, G4VSensitiveDetector* )
{
    printf("G4DAEGeometry::LoadFromGDML need to define -DEXPORT_G4GDML if GDML is available \n");  
    return NULL ;
}
#endif




G4DAEGeometry* G4DAEGeometry::Load(const G4VPhysicalVolume* world)
{
   if( world == NULL ) world = G4TransportationManager::GetTransportationManager()->
             GetNavigatorForTracking()->GetWorldVolume();

   assert(world);
   cout << "G4DAEGeometry::Load " << world->GetName() << endl ; 

   G4DAEGeometry* geo = new G4DAEGeometry();
   G4DAETransformCache* cache = geo->CreateTransformCache(world); 
   cache->Archive(".cache"); 
  
   //
   // where to archive transform cache when getting geometry from MEMORY (ie full NuWa run)? 
   // once have integrated DAE writing into here can write archive as sidecar to the .dae
   //
   return geo ;
}


void G4DAEGeometry::Clear()
{
   m_pvname.clear();
   m_transform.clear();
   m_pvsd.clear() ;
   m_material.clear() ;

   m_pvcount = 0 ;
   m_sdcount = 0 ;
}



void G4DAEGeometry::MakeMaterialMap()
{
    const G4MaterialTable* t = G4Material::GetMaterialTable();
    G4int n = G4Material::GetNumberOfMaterials();
    cout << " G4DAEGeometry::MakeMaterialMap numOfMaterials " << n << endl ;


    for(G4int i=0 ; i < n ; i++)
    {
         G4Material* m = (*t)[i];
         G4int index = m->GetIndex();
         G4String name = m->GetName();

         std::stringstream ss ; 
         ss << index ;  

         m_material[name] = ss.str() ;

         cout << " material " 
              << " i " << i
              << " index " << index 
              << " name"   << name 
              << " sindex " << ss.str()
              << endl; 
    } 
}




G4DAETransformCache* G4DAEGeometry::CreateTransformCache(const G4VPhysicalVolume* world)
{
   if( world == NULL )
       world = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();

   assert(world);
   Clear();


   const G4LogicalVolume* lworld = world->GetLogicalVolume();

   // side effects, split off into own structure
   m_pvname.push_back(world->GetName());         // manual World entry, for indice alignment 
   m_transform.push_back(G4AffineTransform());

   G4DAETransformCache* cache = new G4DAETransformCache();

   PVStack_t pvStack ;                           // topmost volume not on stack
   
   TraverseVolumeTree( lworld, pvStack, cache );

   size_t npv = m_pvname.size() ;
   assert( npv == m_transform.size() );

   cout << "G4DAEGeometry::CreateTransformCache found " << npv << " volumes " << endl ; 

   MakeMaterialMap();
   cache->AddMetadata("MaterialMap", m_material );


   return cache; 
}







void G4DAEGeometry::Dump()
{

   size_t npv = m_pvname.size() ;
   assert( npv == m_transform.size() );

   for( size_t index=0; index < npv; ++index ){ 
       cout << index << " " << transform_rep( m_transform[index] ) << m_pvname[index] << '\n' ;  
   }

   cout << "  (pv,sd) index pairs  " << m_pvsd.size() 
        << "  pvcount " << m_pvcount  
        << "  sdcount " << m_sdcount 
        << endl ;

   for(PVSDMap_t::iterator it = m_pvsd.begin(); it != m_pvsd.end(); it++) 
   {
       cout << " pv" << setw(10) << it->first 
            << " sd" << setw(10) << it->second
            << endl ;
   } 


}

void G4DAEGeometry::AddSensitiveLVName(const std::string& lvname)
{
    m_lvsensitive.push_back(lvname);
}
void G4DAEGeometry::AddSensitiveLVNames(const char* envkey, char delim)
{
    const char* line = getenv(envkey);
    split( m_lvsensitive, line, delim);
}
void G4DAEGeometry::DumpSensitiveLVNames()
{
   for(vector<string>::iterator it=m_lvsensitive.begin() ; it != m_lvsensitive.end() ; it++ ) cout << *it << endl ; 
}


bool G4DAEGeometry::VisitFakeAssignSensitive(G4LogicalVolume* lv, const PVStack_t pvStack, G4VSensitiveDetector* sd)
{
    string lvname = lv->GetName();
    bool senlv = (find( m_lvsensitive.begin(), m_lvsensitive.end(), lvname ) != m_lvsensitive.end() ) ;
    if(!senlv) return false ;

    lv->SetSensitiveDetector(sd);

    size_t indexMax = pvStack.size() - 1;
    G4VPhysicalVolume* pvtop = pvStack[indexMax];
    assert( pvtop->GetLogicalVolume() == lv ); 

    string pvname = pvtop->GetName();
    //cout << setw(40) << lvname << " " << pvname << endl ; 

    //for (size_t index = 0 ; index <= indexMax  ; ++index ) cout << setw(2) << index << " " << pvStack[index]->GetName() << endl ; 
    
    return true;
}

void G4DAEGeometry::FakeAssignSensitive(G4LogicalVolume* lv, PVStack_t pvStack, G4VSensitiveDetector* sd)
{
    bool senlv = VisitFakeAssignSensitive( lv, pvStack, sd );
    int n = lv->GetNoDaughters() ;
    if( senlv ){
         assert( n == 0 ); // not expecting SD to have daughters
    } 

    for (G4int i=0;i<n;i++)   
    {   
        G4VPhysicalVolume* pv = lv->GetDaughter(i);

        PVStack_t pvStackPlus(pvStack);     // copy ctor: each node of the recursion gets its own stack 
        pvStackPlus.push_back(pv);

        FakeAssignSensitive(pv->GetLogicalVolume(), pvStackPlus, sd );
    }   
}






void G4DAEGeometry::TraverseVolumeTree(const G4LogicalVolume* const volumePtr, PVStack_t pvStack, G4DAETransformCache* cache)
{
    // Recursive traverse of all volumes, 
    // performance is irrelevant as run only once 
    //
    // NB position of VisitPV invokation matches that used
    // by G4DAE COLLADA exporter in order for volume order to match
    //

    VisitPV(volumePtr, pvStack, cache );  


    for (G4int i=0;i<volumePtr->GetNoDaughters();i++)   
    {   
        G4VPhysicalVolume* physvol = volumePtr->GetDaughter(i);

        PVStack_t pvStackPlus(pvStack);     // copy ctor: each node of the recursion gets its own stack 
        pvStackPlus.push_back(physvol);

        TraverseVolumeTree(physvol->GetLogicalVolume(),pvStackPlus, cache);
    }   
}



void G4DAEGeometry::VisitPV(const G4LogicalVolume* const volumePtr, const PVStack_t pvStack, G4DAETransformCache* cache)
{
    if(pvStack.size() == 0)
    {
        cout << "G4DAEGeometry::VisitPV skip empty stack " << endl ; 
        return ; 
    }

    G4NavigationHistory navigationHistory ; 
    size_t indexMax = pvStack.size() - 1;
    for (size_t index = 0 ; index <= indexMax  ; ++index ){
         G4VPhysicalVolume* pv = pvStack[index] ; 
         EVolume volumeType = VolumeType(pv); 
         assert( volumeType == kNormal );  // PMTs etc.. not being handled as replicas ?
         navigationHistory.NewLevel( pv, volumeType );  
    }
    G4TouchableHistory touchableHistory(navigationHistory);
    const G4AffineTransform& transform = touchableHistory.GetHistory()->GetTopTransform();
    string name = touchableHistory.GetVolume(0)->GetName();

    G4VSensitiveDetector* sensitive = volumePtr->GetSensitiveDetector();
    if( sensitive )
    {
       std::size_t id = TouchableToIdentifier( touchableHistory );
       //if(id == 0) cout << "G4DAEGeometry::VisitPV " << name << " WARNING SD with no identifier " << endl ; 

       if(id > 0) cache->Add(id, transform) ; 

       m_pvsd[m_pvcount] = m_sdcount ;
       m_sdcount++; 
    }
    m_pvcount++;  
    m_pvname.push_back(name);
    m_transform.push_back(transform);
}





std::size_t G4DAEGeometry::TouchableToIdentifier( const G4TouchableHistory& )
{
    //return m_sdcount + 1;  // override this in detector specialization subclasses 
    return m_pvcount ;  //  pvcount is more useful for debugging comparisons
}




EVolume G4DAEGeometry::VolumeType(G4VPhysicalVolume* pv) const
{
  // from G4 future
  EVolume type;
  EAxis axis;
  G4int nReplicas;
  G4double width,offset;
  G4bool consuming;
  if ( pv->IsReplicated() )
  {
    pv->GetReplicationData(axis,nReplicas,width,offset,consuming);
    type = (consuming) ? kReplica : kParameterised;
  }
  else
  {
    type = kNormal;
  }
  return type;
}



G4AffineTransform* G4DAEGeometry::GetNodeTransform(std::size_t index)
{
   return ( index < m_transform.size() ) ? &m_transform[index] : NULL ; 
}

std::string& G4DAEGeometry::GetNodeName(std::size_t index)
{
   std::size_t idx = ( index < m_pvname.size() ) ? index : 0 ;  
   return m_pvname[idx] ;
}





