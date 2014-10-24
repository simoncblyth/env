#include "GaudiKernel/DeclareFactoryEntries.h"
#include "GaudiKernel/PropertyMgr.h"

#include "G4VPhysicalVolume.hh"
#include "G4TransportationManager.hh"
#include "G4SolidStore.hh"

#ifdef EXPORT_G4GDML
#include "G4GDMLParser.hh"
#endif

#ifdef EXPORT_G4DAE
#include "G4DAEParser.hh"
#endif

#ifdef EXPORT_G4WRL
#include "G4UImanager.hh"
#endif


#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <stdlib.h>  
#include <iostream>

#include "DetDesc/ILVolume.h"
#include "DetDesc/IPVolume.h"
#include "DetDesc/IDetectorElement.h"
#include "DetDesc/IGeometryInfo.h"
#include "DetDesc/DetectorElement.h"

/// Local 
#include "GiGaRunActionExport.h"

// ============================================================================
/** @file 
 *
 *  Implementation file for class : GiGaRunActionExport
 *
 */
// ============================================================================

// Declaration of the Tool Factory
DECLARE_TOOL_FACTORY( GiGaRunActionExport );

/** standard constructor 
 *  @see GiGaPhysListBase
 *  @see GiGaBase 
 *  @see AlgTool 
 *  @param type type of the object (?)
 *  @param name name of the object
 *  @param parent  pointer to parent object
 *
 *  Implementation based on 
 *     external/build/LCG/geant4.9.2.p01/examples/extended/persistency/gdml/G02/src/DetectorConstruction.cc 
 */
GiGaRunActionExport::GiGaRunActionExport
( const std::string& type   ,
  const std::string& name   ,
  const IInterface*  parent ) 
  : GiGaRunActionBase( type , name , parent )
{  
   // m_history = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->CreateTouchableHistory(); 

    declareProperty("PackedIdPropertyName",m_idParameter="PmtID",
                    "The name of the user property holding the PMT ID.");

};

GiGaRunActionExport::~GiGaRunActionExport()
{
};



EVolume GiGaRunActionExport::VolumeType(G4VPhysicalVolume* pv) const
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


static void split(std::vector<std::string>& out, const std::string& src, char del)
{
    out.clear();
    std::string::size_type pos = 0, siz = src.size();
    while (pos < siz) {
        std::string::size_type found = src.find(del,pos);
        if (found == std::string::npos) {
            out.push_back(src.substr(pos,siz-pos));
            break;
        }
        out.push_back(src.substr(pos,found-pos));
        pos = found+1;
    }
}

static std::string dumpstr(const G4TouchableHistory* g4hist)
{
    std::stringstream ss;
    int siz = g4hist->GetHistoryDepth();
    for (int ind=0; ind < siz; ++ind) {
        ss << g4hist->GetVolume(ind)->GetName() << "\n";
    }
    ss << std::ends;
    return ss.str();
}


// simplified version of NuWa-trunk/dybgaudi/Simulation/G4DataHelpers/src/components/TH2DE.h
StatusCode GiGaRunActionExport::GetBestDetectorElement(const G4TouchableHistory* inHistory,
                                                const IDetectorElement* &outElement,
                                               int& outCompatibility)
{
    if (!inHistory->GetHistoryDepth()) {
        warning() << "TH2DE::GetBestDetectorElement given an empty history" << endreq;
        return StatusCode::FAILURE;
    }
    const IDetectorElement* de = NULL ;
    /*
    const IDetectorElement* de = this->CheckCache(inHistory);
    if (de) {
        outElement = de;
        return StatusCode::SUCCESS;
    }
    TouchableHistory_t th;      // for cache
    */
        
    NameHistory_t name_history; // always read backwards, [0] = daughter, [n] = ancestors
    const int depth = inHistory->GetHistoryDepth();
    for (int ind=0; ind < depth; ++ind) {
        G4VPhysicalVolume* g4pv = inHistory->GetVolume(ind);
        //th.push_back(g4pv);

        std::string full_name = g4pv->GetName();

        verbose() << ind << ": " << full_name << endreq;

        std::vector<std::string> names;
        split(names,full_name,'#');

#if 0
        for (size_t iname=0; iname<names.size(); ++iname) {
            debug() << iname << ": [" << names[iname] << "]";
        }
        debug() << endreq;
#endif

        if (depth-ind == 1) { // have /dd/Structure top level DE
            if (names.size() != 1) {
                warning() << "got unknown type at top of history: " << full_name << ", " <<names.size() << endreq;
                return StatusCode::FAILURE;
            }

            DataObject* obj = 0;
            StatusCode sc = detSvc()->retrieveObject(names[0],obj);
            if (sc.isFailure()) {
                warning() << "failed to get DetectorElement: " << names[0] << endreq;
                return StatusCode::FAILURE;
            }
            de = dynamic_cast<const IDetectorElement*>(obj);
            if (!de) {
                warning() << "failed to dynamic_cast<DetectorElement*>: " << names[0] << endreq;
                return StatusCode::FAILURE;
            }
            break;
        }
            

        if (names.size() > 2) {   // have pv1#pv2#pv3 style path
            DataObject* obj = 0;
            StatusCode sc = detSvc()->retrieveObject(names[0],obj);
            if (sc.isFailure()) {
                warning() << "failed to get LVolume: " << names[0] << endreq;
                return StatusCode::FAILURE;
            }
            const ILVolume* lv = dynamic_cast<const ILVolume*>(obj);
            if (!lv) {
                warning() << "failed to dynamic_cast<LVolume*>: " << names[0] << endreq;
                return StatusCode::FAILURE;
            }
            const IPVolume* pv = (*lv)[names[1]];
            lv = pv->lvolume();
            NameHistory_t reverse;
            for (std::string::size_type iname = 2; iname < names.size(); ++iname) {
                pv = (*lv)[names[iname]];
                reverse.push_back(LvPvPair_t(lv->name(),pv->name()));
                lv = pv->lvolume();
            }
            name_history.insert(name_history.end(),reverse.rbegin(),reverse.rend());
        }

        name_history.push_back(LvPvPair_t(names[0],names[1]));

    } // loop over history

    if (!de) {
        const int depth = inHistory->GetHistoryDepth();
        warning() << "failed to find top level supporting DetectorElement, history has depth of " << depth << "\n";
        for (int ind=0; ind < depth; ++ind) {
            G4VPhysicalVolume* g4pv = inHistory->GetVolume(ind);
            warning() << "\n\t(" << ind << ") " << g4pv->GetName();
        }
        warning() << endreq;
        return StatusCode::FAILURE;
    }

    de = this->FindDE(de,name_history);
    if (!de) {
        warning() << "failed to find DetectorElement for TouchableHistory:\n" << dumpstr(inHistory) << endreq;
        return StatusCode::FAILURE;
    }

    //m_THcache[th] = de;
    outElement = de;
    outCompatibility = name_history.size();
    return StatusCode::SUCCESS;
}

int GiGaRunActionExport::InHistory(const IDetectorElement* de, const NameHistory_t& name_history)
{
    const IGeometryInfo* gi = de->geometry();
    if (!gi->hasSupport()) return -1;

    const ILVolume::ReplicaPath& rpath = gi->supportPath();
    const IGeometryInfo* support_gi = gi->supportIGeometryInfo();
    const ILVolume* lv = support_gi->lvolume();

    verbose() << "InHistory de=" << de->name() << endreq;

    // Walk the DE's support and match against name_history;
    size_t index = name_history.size();
    for (size_t ind = 0; index && ind < rpath.size(); ++ind) {
        IPVolume* pv = lv->pvolumes()[rpath[ind]];

        --index;
        const LvPvPair_t& check = name_history[index];

        verbose() << "("<<index<<") lvpv=" << lv->name() << "#" << pv->name() 
                  << " =?= " 
                  << check.first << "#" << check.second << endreq;

        if (lv->name() != check.first) return -1;
        if (pv->name() != check.second) return -1;

        lv = pv->lvolume();
    }
    return index;
}


const IDetectorElement* GiGaRunActionExport::FindChildDE(const IDetectorElement* de, NameHistory_t& name_history)
{
    IDetectorElement::IDEContainer& children = de->childIDetectorElements();
    size_t nchildren = children.size();
    verbose() << "Finding children from " << nchildren << endreq;
    for (size_t ichild = 0; ichild < nchildren; ++ichild) {
        IDetectorElement* child = children[ichild];
        int index = this->InHistory(child,name_history);

        if (index<0) continue;

        verbose() << "Found child: " << child->name() << " at index " <<index<<" lv#pv=" << name_history[index].first << "#"<< name_history[index].second << endreq;

        // strip off used history
        while ((int)name_history.size() > index) {
            LvPvPair_t lvpv = name_history.back();
            verbose () << "\tpoping: index="<<index<<", size=" << name_history.size() << " [" << lvpv.first << "#" << lvpv.second << "]" << endreq;
            name_history.pop_back();
        }
        return child;
    }
    return 0;
}


const IDetectorElement* GiGaRunActionExport::FindDE(const IDetectorElement* de, NameHistory_t& name_history)
{
    // If exhausted the NH then the current DE must be the one
    if (!name_history.size()) return de;

#if 0
    debug() << "FindDE: " << de->name() << endreq;
    for (size_t inh=0; inh<name_history.size(); ++inh) {
        debug() << inh <<": " << name_history[inh].first << "#" << name_history[inh].second << endreq;
    }
#endif

    LvPvPair_t lvpv(name_history.back().first,name_history.back().second);

    std::string de_lvname = de->geometry()->lvolumeName();
    if (de_lvname != lvpv.first) {
        warning() << "The given DE's LV does not match LV from top of history: "
                  << de_lvname <<" != "<< lvpv.first << endreq;
        return 0;
    }

    // Find immediate child that points into the history
    const IDetectorElement* child_de = this->FindChildDE(de,name_history);

    // If one found, recurse.
    if (child_de) return this->FindDE(child_de,name_history);

    // If we get here, we have reached the end of possible DEs.
    // Pop off the touchable history coresponding to current DE and return;
    name_history.pop_back();
    return de;
}


void GiGaRunActionExport::VisitPV( const PVStack_t& pvStack )
{
    /*
        #. a single PV alone is not enough for identification, 
           but a PV from a full traverse there is an implicit(or now explicit) stack 
           of PV, which succeeds to precisely locate a volume


    How to create G4TouchableHistory instances from PV stacks 
    obtained by full recursive traverse is not documented. Usually 
    these are created from a global position in ProcessHits.

    The below is a guess, TO BE VALIDATED.

    */

    std::cout << "VisitPV " << pvStack.size() << std::endl ; 
    if(pvStack.size() == 0)
    {
        std::cout << "VisitPV skip empty stack " << std::endl ; 
        return ; 
    }

    G4NavigationHistory navigationHistory ; 

    size_t indexMax = pvStack.size() - 1;
    for (size_t index = 0 ; index <= indexMax  ; ++index ){

         G4VPhysicalVolume* pv = pvStack[index] ; 

         std::cout << std::setw(2) << index 
                   << " " << pv 
                   << " copyNo " << std::setw(5) << pv->GetCopyNo() 
                   << " " << pv->GetName() 
                   << std::endl  ;

         EVolume volumeType = VolumeType(pv); 
         assert( volumeType == kNormal );  // PMTs etc.. not being handled as replicas ?
         navigationHistory.NewLevel( pv, volumeType );  
    }

    std::cout << "NavigationHistory TopVolumeName " << navigationHistory.GetTopVolume()->GetName() << std::endl ;  

    G4TouchableHistory touchableHistory(navigationHistory);
    std::cout << "TouchableHistory " << touchableHistory.GetHistoryDepth() << std::endl ;

    const DetectorElement* de = this->SensDetElem(touchableHistory);
    int pmtid = 0 ; 
    if (de){
        pmtid = this->SensDetId(*de);
    }

    std::string name = touchableHistory.GetVolume(0)->GetName();
    const G4AffineTransform& transform = touchableHistory.GetHistory()->GetTopTransform();

    m_pvid.push_back(pmtid);
    m_pvname.push_back(name);
    m_transform.push_back(transform);

    std::cout << "PMTID " << pmtid << " 0x" << std::setw(7) << std::hex << pmtid << std::dec << " " << name << std::endl ;
}


// Return the SensitiveDetector ID for the given touchable history.
// from DsPmtSensDet::SensDetElem
const DetectorElement* GiGaRunActionExport::SensDetElem(const G4TouchableHistory& hist)
{
    const IDetectorElement* idetelem = 0;
    int steps=0;

    if (!hist.GetHistoryDepth()) {
        error() << "DsPmtSensDet::SensDetElem given empty touchable history" << endreq;
        return 0;
    }

    StatusCode sc = 
        this->GetBestDetectorElement(&hist,idetelem,steps);
    if (sc.isFailure()) {      // verbose warning
        warning() << "Failed to find detector element in:\n";
        warning() << "\tfor touchable history:\n";
        for (int ind=0; ind < hist.GetHistoryDepth(); ++ind) {
            warning() << "\t (" << ind << ") " 
                      << hist.GetVolume(ind)->GetName() << "\n";
        }
        warning() << endreq;
        return 0;
    }
    return dynamic_cast<const DetectorElement*>(idetelem);
}

// from DsPmtSensDet::SensDetId
int  GiGaRunActionExport::SensDetId(const DetectorElement& de)
{
    const DetectorElement* detelem = &de;
    while (detelem) {
        if (detelem->params()->exists(m_idParameter)) {
            break;
        }
        detelem = dynamic_cast<const DetectorElement*>(detelem->parentIDetectorElement());
    }
    if (!detelem) {
        warning() << "Could not get PMT detector element starting from " << de << endreq;
        return 0;
    }
    return detelem->params()->param<int>(m_idParameter);
}


void GiGaRunActionExport::TraverseVolumeTree(const G4LogicalVolume* const volumePtr, PVStack_t pvStack)
{
    //performance is irrelevant as run once only per geometry 
    // CAUTION: where VisitPV is called to match 
    // PV ordering used in COLLADA exporter

    VisitPV( pvStack );  

    for (G4int i=0;i<volumePtr->GetNoDaughters();i++)   // Traverse all the children!
    {   
        G4VPhysicalVolume* physvol = volumePtr->GetDaughter(i);

        PVStack_t pvStackPlus(pvStack);     // copy ctor: each node of the recursion gets its own stack 
        pvStackPlus.push_back(physvol);

        TraverseVolumeTree(physvol->GetLogicalVolume(),pvStackPlus);
    }   
}


void GiGaRunActionExport::WriteIdMap(G4VPhysicalVolume* wpv, const G4String& path )
{
   // collect identifiers from full traverse, 
   // many placeholder zeros expected

   std::cout << "GiGaRunActionExport::WriteIdMap to " << path 
             << " WorldVolume : " << wpv->GetName() 
             << std::endl ; 

   const G4LogicalVolume* lvol = wpv->GetLogicalVolume();

   m_pvid.clear();
   m_pvname.clear();
   m_transform.clear();

   // manual World entry, for indice alignment 
   m_pvid.push_back(0);
   m_pvname.push_back(wpv->GetName());
   m_transform.push_back(G4AffineTransform());

   PVStack_t pvStack ;     // Universe not on stack
   TraverseVolumeTree( lvol, pvStack );

   size_t npv = m_pvid.size() ;
   assert( npv == m_pvname.size() );
   assert( npv == m_transform.size() );

   std::ofstream fp;
   fp.open(path);

   fp << "# GiGaRunActionExport::WriteIdMap fields: index,pmtid,pmtid(hex),pvname  npv:" << npv << '\n' ;  

   for( size_t index=0; index < npv; ++index ){ 
       int id = m_pvid[index] ;  
       G4AffineTransform& transform = m_transform[index];
      
       G4RotationMatrix rotation = transform.NetRotation();
       G4ThreeVector rowX = rotation.rowX();
       G4ThreeVector rowY = rotation.rowY();
       G4ThreeVector rowZ = rotation.rowZ();

       G4ThreeVector translation = transform.NetTranslation(); 


       //  12224 0 0  (-794970,-154105,7260) (-0.209619,0.977783,0)(-0.977783,-0.209619,0)(0,0,1) /dd/.... 

       std::string name = m_pvname[index] ;    // for debug, NOT identity matching 
       fp << index << " " 
          << id << " " 
          << std::hex << id << std::dec << " " 
          << translation << " "
          << rowX << rowY << rowZ << " "  //  
          << name 
          << '\n' ;  
   }
   fp.close();

}


void GiGaRunActionExport::WriteGDML(G4VPhysicalVolume* wpv, const G4String& path )
{
#ifdef EXPORT_G4GDML
   if(path.length() == 0 || wpv == 0){
       std::cout << "GiGaRunActionExport::WriteGDML invalid path OR NULL PV  " << path << std::endl ;
       return ;  
   }
   std::cout << "GiGaRunActionExport::WriteGDML to " << path << std::endl ;
   G4GDMLParser parser ;
   parser.Write(path, wpv);
#else
   std::cout << "GiGaRunActionExport::WriteGDML BUT this installation  not compiled with -DEXPORT_G4GDML " << std::endl ; 
#endif
}

void GiGaRunActionExport::WriteDAE(G4VPhysicalVolume* wpv, const G4String& path, G4bool recreatePoly  )
{
#ifdef EXPORT_G4DAE
   if(path.length() == 0 || wpv == 0){
       std::cout << "GiGaRunActionExport::WriteDAE invalid path OR NULL PV  " << path << std::endl ;
       return ;  
   }
   std::cout << "GiGaRunActionExport::WriteDAE to " << path << " recreatePoly " << recreatePoly << std::endl ;
   G4DAEParser parser ;
   G4bool refs = true ; 
   G4int nodeIndex = -1 ;   // so World is volume 0 
   parser.Write(path, wpv, refs, recreatePoly, nodeIndex );
#else
   std::cout << "GiGaRunActionExport::WriteDAE BUT this installation  not compiled with -DEXPORT_G4DAE " << std::endl ; 
#endif
}

G4bool GiGaRunActionExport::FileExists(const char *fileName){
    std::ifstream infile(fileName);
    return infile.good();
}

G4String GiGaRunActionExport::FilePath( const G4String& base , G4int index, const G4String& ext , G4bool wantfree )
{
   /*
         base 
              prefix path eg "/path/to/directory/g4_" "g4_" 
         counter
              0,1,..,99 
         ext 
              ".gdml"
         wantfree
              if true return a blank string if the path is already existing
              if false return path regardless of existance 
             
   */
   std::ostringstream ss;
   ss << base << std::setw(2) << std::setfill('0') << index << ext << std::setfill(' ') ;
   std::string path = ss.str();
   G4bool exists_ = FileExists(path.c_str());
   if( wantfree && exists_ ){
        //std::cout << "path " << path << " exists already " << std::endl ;
        return G4String("") ; 
   } 
   return G4String(path);
}

G4String GiGaRunActionExport::FreeFilePath( const G4String& base, const G4String& ext )
{
    G4int imax(99);
    G4int i(0);
    G4String path("") ; 
    while(path.length() == 0){
        path = FilePath( base,  i++, ext , true );
        if( i == imax ) break ; 
    }
    std::cout << "FreeFilePath  return " << path << " i " << i <<  std::endl ;
    return path ; 
}

void GiGaRunActionExport::CleanSolidStore()
{
    std::cout << "GiGaRunActionExport::CleanSolidStore deleting all solids from the store " <<  std::endl ;
    G4SolidStore::Clean();
}


void GiGaRunActionExport::InitVis(const char* /*driver*/)
{
#ifdef EXPORT_G4WRL
   G4UImanager* ui = G4UImanager::GetUIpointer() ; 
   ui->ApplyCommand("/vis/open VRML2FILE");
   ui->ApplyCommand("/vis/geometry/list all");
   ui->ApplyCommand("/vis/viewer/set/culling global false");
   ui->ApplyCommand("/vis/viewer/set/culling coveredDaughters false");
   //ui->ApplyCommand("/vis/viewer/set/lineSegmentsPerCircle 100");    
#else
   std::cout << "GiGaRunActionExport::InitVis BUT this installation  not compiled with -DEXPORT_G4WRL " << std::endl ; 
#endif
}

void GiGaRunActionExport::FlushVis(const char* /*driver*/)
{
#ifdef EXPORT_G4WRL
   G4UImanager* ui = G4UImanager::GetUIpointer() ; 
   ui->ApplyCommand("/vis/drawVolume");
   ui->ApplyCommand("/vis/viewer/flush");
#else
   std::cout << "GiGaRunActionExport::FlushVis BUT this installation  not compiled with -DEXPORT_G4WRL " << std::endl ; 
#endif
}


void GiGaRunActionExport::WriteVis(const char* /*driver*/)
{
#ifdef EXPORT_G4WRL
   G4UImanager* ui = G4UImanager::GetUIpointer() ; 
   G4cout << "GiGaRunActionExport::WriteVis vis open " << G4endl ; 
   ui->ApplyCommand("/vis/open VRML2FILE");
   G4cout << "GiGaRunActionExport::WriteVis list geom " << G4endl ; 
   ui->ApplyCommand("/vis/geometry/list all");
   G4cout << "GiGaRunActionExport::WriteVis set culling 1  " << G4endl ; 
   ui->ApplyCommand("/vis/viewer/set/culling global false");
   G4cout << "GiGaRunActionExport::WriteVis set culling 2  " << G4endl ; 
   ui->ApplyCommand("/vis/viewer/set/culling coveredDaughters false");
   //ui->ApplyCommand("/vis/viewer/set/lineSegmentsPerCircle 100");    
   G4cout << "GiGaRunActionExport::WriteVis drawVolume  " << G4endl ; 
   ui->ApplyCommand("/vis/drawVolume");
   G4cout << "GiGaRunActionExport::WriteVis flush  " << G4endl ; 
   ui->ApplyCommand("/vis/viewer/flush");
   G4cout << "GiGaRunActionExport::WriteVis done  " << G4endl ; 
#else
   std::cout << "GiGaRunActionExport::WriteVis BUT this installation  not compiled with -DEXPORT_G4WRL " << std::endl ; 
#endif
}

G4String GiGaRunActionExport::GetEnv( const char* envvar , const char* def )
{
   char const* tmp = getenv(envvar);   // no trailing slash 
   G4String val = ( tmp == NULL ) ? def : tmp ;  
   return val ; 
}

void GiGaRunActionExport::AbruptExit()
{
   std::cout << "GiGaRunActionExport::AbruptExit due to G4DAE_EXPORT_EXIT: " << std::endl ;  
   exit(0);
}


void GiGaRunActionExport::BeginOfRunAction( const G4Run* run )
{

  if( 0 == run ) 
    { Warning("BeginOfRunAction:: G4Run* points to NULL!") ; }

   G4VPhysicalVolume* wpv = G4TransportationManager::GetTransportationManager()->
      GetNavigatorForTracking()->GetWorldVolume();


   // setup output directories
   G4String xdir = GetEnv("G4DAE_EXPORT_DIR", ".");   // no trailing slash 
   G4String vdir(xdir);
   vdir += "/" ;       // VRML needs a trailing slash 
   setenv("G4VRMLFILE_DEST_DIR", vdir.c_str(), 0 );

   G4String base(xdir);
   base += "/g4_" ; 

   // write geometry, multiple times and interleaved for DAE/WRL interference testing 
   G4String xseq = GetEnv("G4DAE_EXPORT_SEQUENCE","VGD");
   const char* seq = xseq.c_str();

   for (int i = 0; i < strlen(seq); i++){
       char c = seq[i];
       std::cout << "GiGaRunActionExport::BeginOfRunAction i " << i << " c " << c << std::endl ;  
       switch (c) 
       {
          case 'M':
                 WriteIdMap( wpv, FreeFilePath(base, ".idmap"));
                 break;
           case 'V':
                 WriteVis("VRML2FILE");
                 break;
          case 'I':
                 InitVis("VRML2FILE");
                 break;
          case 'F':
                 FlushVis("VRML2FILE");
                 break;
          case 'G':
                 WriteGDML( wpv, FreeFilePath(base, ".gdml"));
                 break;
          case 'A':
                 WriteDAE( wpv, FreeFilePath(base, ".dae"), true );
                 break;
          case 'D':
                 WriteDAE( wpv, FreeFilePath(base, ".dae"), false );
                 break;
          case 'C':
                 CleanSolidStore();
                 break;
          case 'X':
                 AbruptExit();
                 break;
       }
   }

  
};

void GiGaRunActionExport::EndOfRunAction( const G4Run* run )
{
  if( 0 == run ) 
    { Warning("EndOfRunAction:: G4Run* points to NULL!") ; }
};


