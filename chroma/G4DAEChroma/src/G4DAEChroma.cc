#include "G4DAEChroma/G4DAEChroma.hh"

#ifdef WITH_CHROMA_ZMQ
#include "Chroma/ChromaPhotonList.hh"
#include "ZMQRoot/ZMQRoot.hh"
#endif

#include "G4VProcess.hh"
#include "G4AffineTransform.hh"
#include "G4TransportationManager.hh"
#include "G4NavigationHistory.hh"
#include "G4TouchableHistory.hh"


G4DAEChroma* G4DAEChroma::fG4DAEChroma = 0;

G4DAEChroma* G4DAEChroma::GetG4DAEChroma()
{
  if(!fG4DAEChroma)
  {
     fG4DAEChroma = new G4DAEChroma;
  }
  return fG4DAEChroma;
}

G4DAEChroma* G4DAEChroma::GetG4DAEChromaIfExists()
{ 
  return fG4DAEChroma ;
}



G4DAEChroma::G4DAEChroma() :
    fZMQRoot(0),
    fPhotonList(0),
    fPhotonList2(0),
    m_transform_cache_created(false)
{ 
#ifdef WITH_CHROMA_ZMQ
  fPhotonList = new ChromaPhotonList;   
  fZMQRoot = new ZMQRoot("CSA_CLIENT_CONFIG");  //TODO: pass along this config from upper ctor ? change default G4DAECHROMA_CLIENT_CONFIG
#endif
}

G4DAEChroma::~G4DAEChroma()
{
#ifdef WITH_CHROMA_ZMQ
   if(fPhotonList)  delete fPhotonList ; 
   if(fPhotonList2) delete fPhotonList2 ; 
   if(fZMQRoot)     delete fZMQRoot ; 
#endif
}


void G4DAEChroma::ClearAll()
{
#ifdef WITH_CHROMA_ZMQ
 if(fPhotonList){ 
      G4cout<< "::ClearAll fPhotonList  "<<G4endl;
      fPhotonList->ClearAll(); 
  }
  if(fPhotonList2){ 
      G4cout<< "::ClearAll fPhotonList2 "<<G4endl;
      fPhotonList2->ClearAll(); 
  } 
#endif
}

void G4DAEChroma::CollectPhoton(const G4Track* aPhoton )
{
#ifdef WITH_CHROMA_ZMQ
   G4ParticleDefinition* pd = aPhoton->GetDefinition();
   assert( pd->GetParticleName() == "opticalphoton" );

   G4String pname="-";
   const G4VProcess* process = aPhoton->GetCreatorProcess();
   if(process) pname = process->GetProcessName();
   G4cout << " OP : " 
          << " ProcessName " << pname 
          << " ParentID "    << aPhoton->GetParentID() 
          << " TrackID "     << aPhoton->GetTrackID() 
          << " KineticEnergy " << aPhoton->GetKineticEnergy() 
          << " TotalEnergy " << aPhoton->GetTotalEnergy() 
          << " TrackStatus " << aPhoton->GetTrackStatus() 
          << " CurrentStepNumber " << aPhoton->GetCurrentStepNumber() 
          << G4endl;

   assert( pname == "Cerenkov" || pname == "Scintillation" );

   G4ThreeVector pos = aPhoton->GetPosition()/mm ;
   G4ThreeVector dir = aPhoton->GetMomentumDirection() ;
   G4ThreeVector pol = aPhoton->GetPolarization() ;
   float time = aPhoton->GetGlobalTime()/ns ;
   float wavelength = (h_Planck * c_light / aPhoton->GetKineticEnergy()) / nanometer ;

   fPhotonList->AddPhoton( 
              pos.x(), pos.y(), pos.z(),
              dir.x(), dir.y(), dir.z(),
              pol.x(), pol.y(), pol.z(), 
              time, 
              wavelength );

#endif
}

void G4DAEChroma::Propagate(G4int batch_id)
{
#ifdef WITH_CHROMA_ZMQ

  // defer creating transform cache, to ensure geometry in place
  if(!m_transform_cache_created){
       CreateTransformCache();
       //DumpTransformCache();       
  }  

  fPhotonList->SetUniqueID(batch_id);
  G4cout << "::Propagate fPhotonList " <<  G4endl ;   
  fPhotonList->Print(); 
  std::size_t size = fPhotonList->GetSize(); 

  if(size > 0)
  {
      G4cout << "::SendObject " <<  G4endl ;   
      fZMQRoot->SendObject(fPhotonList);
      G4cout << "::ReceiveObject, waiting... " <<  G4endl;   
      fPhotonList2 = (ChromaPhotonList*)fZMQRoot->ReceiveObject();
      G4cout << "::fPhotonList2 " <<  G4endl ;   
      fPhotonList2->Print();
      
      for( std::size_t index = 0 ; index < size ; index++ )
      {
          ProcessHitsChroma( fPhotonList2,  index );
      }   

  } 
  else 
  { 
      G4cout << "::Propagate Skip send/recv for empty CPL " <<  G4endl;   
  }
#else
      G4cout << "::Propagate : NEED TO RECOMPILE USING : -DWITH_CHROMA_ZMQ  " <<  G4endl;   
#endif
}

bool G4DAEChroma::ProcessHitsChroma( const ChromaPhotonList* cpl, std::size_t index )
{
#ifdef WITH_CHROMA_ZMQ
    Hit hit ; 
    cpl->GetPhoton( index, hit.gpos, hit.gdir, hit.gpol, hit.t, hit.wavelength, hit.pmtid );    
    hit.volumeindex = 0 ; //dummy
    hit.LocalTransform(GetNodeTransform(hit.volumeindex));
   
    G4cout << " index " << index 
           << " gpos "  << hit.gpos 
           << " gdir "  << hit.gdir 
           << " gpol "  << hit.gpol 
           << " t "    << hit.t 
           << " wavelength " << hit.wavelength 
           << " pmtid " << hit.pmtid << G4endl ; 

 /*
    DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
    sphit->setHitTime(t);    // Time since event created
    sphit->setSensDetId(pmtid);
    sphit->setLocalPos(lpos); 
    sphit->setPol(lpol);
    sphit->setDir(ldir);
    sphit->setWavelength(wavelength);
    sphit->setType(0);
    sphit->setWeight(weight);
    this->StoreHit(sphit,trackid);
    G4cout << "Stored photon " << trackid << " weight " << weight << " pmtid " << (void*)pmtid << " wavelength(nm) " << wavelength/CLHEP::nm << G4endl;

  */
#endif
    return true;
}



void G4DAEChroma::CreateTransformCache(G4VPhysicalVolume* wpv)
{
   if( wpv == NULL )
   {
       G4cout << "G4DAEChroma::CreateTransformCache trying to access world volume " << G4endl ; 
       wpv = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
   }

   if(wpv == NULL){
       G4cout << "G4DAEChroma::CreateTransformCache ABORT : failed to access WorldVolume " << G4endl ; 
       return ;
   } 

   const G4LogicalVolume* lvol = wpv->GetLogicalVolume();

   m_pvname.clear();
   m_transform.clear();

   // manual World entry, for indice alignment 
   m_pvname.push_back(wpv->GetName());
   m_transform.push_back(G4AffineTransform());

   PVStack_t pvStack ;  // topmost volume not on stack
   TraverseVolumeTree( lvol, pvStack );

   size_t npv = m_pvname.size() ;
   assert( npv == m_transform.size() );

   m_transform_cache_created = true ; 
   G4cout << "G4DAEChroma::CreateTransformCache found " << npv << " volumes " << G4endl ; 
}


void G4DAEChroma::DumpTransformCache()
{

   if(!m_transform_cache_created){
      G4cout << "G4DAEChroma::DumpTransformCache SKIP : not created yet  " << G4endl ; 
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

       std::cout << index << " " 
                 << translation << " "
                 << rowX << rowY << rowZ << " "  
                 << m_pvname[index] 
                 << '\n' ;  
   }
}


void G4DAEChroma::TraverseVolumeTree(const G4LogicalVolume* const volumePtr, PVStack_t pvStack)
{
    // Recursive traverse of all volumes, 
    // performance is irrelevant as run only once 
    //
    // NB position of VisitPV invokation matches that used
    // by G4DAE COLLADA exporter in order for volume order to match

    VisitPV( pvStack );  

    for (G4int i=0;i<volumePtr->GetNoDaughters();i++)   
    {   
        G4VPhysicalVolume* physvol = volumePtr->GetDaughter(i);

        PVStack_t pvStackPlus(pvStack);     // copy ctor: each node of the recursion gets its own stack 
        pvStackPlus.push_back(physvol);

        TraverseVolumeTree(physvol->GetLogicalVolume(),pvStackPlus);
    }   
}

void G4DAEChroma::VisitPV( const PVStack_t& pvStack )
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


EVolume G4DAEChroma::VolumeType(G4VPhysicalVolume* pv) const
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



G4AffineTransform& G4DAEChroma::GetNodeTransform(std::size_t index)
{
   std::size_t idx = ( index < m_transform.size() ) ? index : 0 ;
   return m_transform[idx] ;
}

std::string& G4DAEChroma::GetNodeName(std::size_t index)
{
   std::size_t idx = ( index < m_pvname.size() ) ? index : 0 ;  
   return m_pvname[idx] ;
}




