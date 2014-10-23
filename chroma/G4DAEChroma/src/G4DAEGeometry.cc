#include "G4DAEChroma/G4DAEGeometry.hh"

#include "G4AffineTransform.hh"
#include "G4TransportationManager.hh"
#include "G4NavigationHistory.hh"
#include "G4TouchableHistory.hh"

#ifdef EXPORT_G4GDML
#include "G4GDMLParser.hh"
#endif

#include <stdlib.h>    
#include <iostream>    

using namespace std ; 


G4DAEGeometry::G4DAEGeometry() :
    m_transform_cache_created(false), m_pvcount(0), m_sdcount(0)
{ 
}

G4DAEGeometry::~G4DAEGeometry()
{
}

bool G4DAEGeometry::CacheExists(){
   return m_transform_cache_created ;
}


G4DAEGeometry* G4DAEGeometry::MakeGeometry( const char* geometry )
{
   return ( strcmp(geometry,"MEMORY") == 0 ) ? Load(NULL) : LoadFromGDML(geometry) ;
}


G4DAEGeometry* G4DAEGeometry::LoadFromGDML( const char* geokey )
{
   const char* geopath = getenv(geokey);
   if(geopath == NULL ){
      printf("G4DAEGeometry::LoadFromGDML geokey %s : missing : use \"export-;export-export\" to define  \n", geokey );
      return NULL;
   }   
   printf("geokey %s geopath %s \n", geokey, geopath ); 


   G4VPhysicalVolume* world = NULL ;

#ifdef EXPORT_G4GDML
   G4GDMLParser fParser ; 
   fParser.Read(geopath,false);
   world = fParser.GetWorldVolume();       
#else
   printf("G4DAEGeometry::LoadFromGDML need to define -DEXPORT_G4GDML if GDML is available \n");  
#endif

   return G4DAEGeometry::Load(world);
}

G4DAEGeometry* G4DAEGeometry::Load(const G4VPhysicalVolume* world)
{
   if( world == NULL )
   {
       world = G4TransportationManager::GetTransportationManager()->
             GetNavigatorForTracking()->GetWorldVolume();
   }
   assert(world);
   cout << "G4DAEGeometry::Load " << world->GetName() << endl ; 

   G4DAEGeometry* geo = new G4DAEGeometry();
   geo->CreateTransformCache(world); 
   //geo->DumpTransformCache(); 
 
   return geo ;
}



void G4DAEGeometry::CreateTransformCache(const G4VPhysicalVolume* wpv)
{
   if( wpv == NULL )
   {
       cout << "G4DAEGeometry::CreateTransformCache trying to access world volume " << endl ; 
       wpv = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
   }

   if(wpv == NULL){
       cout << "G4DAEGeometry::CreateTransformCache ABORT : failed to access WorldVolume " << endl ; 
       return ;
   } 

   const G4LogicalVolume* lvol = wpv->GetLogicalVolume();

   m_pvname.clear();
   m_transform.clear();
   m_pvcount = 0 ;
   m_sdcount = 0 ;
   m_pvsd.clear() ;

   // manual World entry, for indice alignment 
   m_pvname.push_back(wpv->GetName());
   m_transform.push_back(G4AffineTransform());

   PVStack_t pvStack ;  // topmost volume not on stack
   TraverseVolumeTree( lvol, pvStack );

   size_t npv = m_pvname.size() ;
   assert( npv == m_transform.size() );

   m_transform_cache_created = true ; 
   cout << "G4DAEGeometry::CreateTransformCache found " << npv << " volumes " << endl ; 

   cout << "  (pv,sd) index pairs  " << m_pvsd.size() 
        << "  pvcount " << m_pvcount  
        << "  sdcount " << m_sdcount 
        << endl ;

   for(PVSDMap::iterator it = m_pvsd.begin(); it != m_pvsd.end(); it++) {
       cout << "pv" << setw(10) << it->first 
            << "sd" << setw(10) << it->second
            << endl ;
   } 

}


void G4DAEGeometry::DumpTransformCache()
{

   if(!m_transform_cache_created){
      G4cout << "G4DAEGeometry::DumpTransformCache SKIP : not created yet  " << G4endl ; 
      return ; 
   } 

   size_t npv = m_pvname.size() ;
   assert( npv == m_transform.size() );

   for( size_t index=0; index < npv; ++index ){ 
       G4AffineTransform& transform = m_transform[index];
      
       G4RotationMatrix rotation = transform.NetRotation();
       G4ThreeVector rowX = rotation.rowX();
       G4ThreeVector rowY = rotation.rowY();
       G4ThreeVector rowZ = rotation.rowZ();

       G4ThreeVector translation = transform.NetTranslation(); 

       cout << index << " " 
                 << translation << " "
                 << rowX << rowY << rowZ << " "  
                 << m_pvname[index] 
                 << '\n' ;  
   }
}


void G4DAEGeometry::TraverseVolumeTree(const G4LogicalVolume* const volumePtr, PVStack_t pvStack)
{
    // Recursive traverse of all volumes, 
    // performance is irrelevant as run only once 
    //
    // NB position of VisitPV invokation matches that used
    // by G4DAE COLLADA exporter in order for volume order to match
    //
    //
    // SD assignments do not survive GDML-ization  
    //

    VisitPV( pvStack );  


    G4VSensitiveDetector* sd = volumePtr->GetSensitiveDetector();

    if( sd ){
       //cout << "SD " << volumePtr->GetName() << endl ; 
       m_pvsd[m_pvcount] = m_sdcount ; 
       m_sdcount++; 
    }
    m_pvcount++;  


    for (G4int i=0;i<volumePtr->GetNoDaughters();i++)   
    {   
        G4VPhysicalVolume* physvol = volumePtr->GetDaughter(i);

        PVStack_t pvStackPlus(pvStack);     // copy ctor: each node of the recursion gets its own stack 
        pvStackPlus.push_back(physvol);

        TraverseVolumeTree(physvol->GetLogicalVolume(),pvStackPlus);
    }   
}

void G4DAEGeometry::VisitPV( const PVStack_t& pvStack )
{

    //std::cout << "VisitPV " << pvStack.size() << std::endl ; 
    if(pvStack.size() == 0)
    {
        std::cout << "VisitPV skip empty stack " << std::endl ; 
        return ; 
    }

    G4NavigationHistory navigationHistory ; 

    size_t indexMax = pvStack.size() - 1;
    for (size_t index = 0 ; index <= indexMax  ; ++index ){

         G4VPhysicalVolume* pv = pvStack[index] ; 

         /*
         std::cout << std::setw(2) << index 
                   << " " << pv 
                   << " copyNo " << std::setw(5) << pv->GetCopyNo() 
                   << " " << pv->GetName() 
                   << std::endl  ;
         */
         EVolume volumeType = VolumeType(pv); 
         assert( volumeType == kNormal );  // PMTs etc.. not being handled as replicas ?
         navigationHistory.NewLevel( pv, volumeType );  
    }

    //std::cout << "NavigationHistory TopVolumeName " << navigationHistory.GetTopVolume()->GetName() << std::endl ;  

    G4TouchableHistory touchableHistory(navigationHistory);
    //std::cout << "TouchableHistory " << touchableHistory.GetHistoryDepth() << std::endl ;

    std::string name = touchableHistory.GetVolume(0)->GetName();
    const G4AffineTransform& transform = touchableHistory.GetHistory()->GetTopTransform();

    m_pvname.push_back(name);
    m_transform.push_back(transform);

    //std::cout << "pv " << name << std::endl ;
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



G4AffineTransform& G4DAEGeometry::GetNodeTransform(std::size_t index)
{
   std::size_t idx = ( index < m_transform.size() ) ? index : 0 ;
   return m_transform[idx] ;
}

std::string& G4DAEGeometry::GetNodeName(std::size_t index)
{
   std::size_t idx = ( index < m_pvname.size() ) ? index : 0 ;  
   return m_pvname[idx] ;
}





