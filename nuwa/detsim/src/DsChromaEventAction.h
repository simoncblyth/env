#ifndef DSCHROMAEVENTACTION_H 
#define DSCHROMAEVENTACTION_H 1

#include "GaudiAlg/GaudiTool.h"
#include "GiGa/GiGaEventActionBase.h"

#include "G4DAEChroma/G4DAEMap.hh"  


/** @class DsChromaEventAction 
 *  
 *  A concrete Event Action. 
 *
 *  @author Simon Blyth
 *  @date   
 */

class DsChromaEventAction : public virtual GiGaEventActionBase
{
public:

  void BeginOfEventAction ( const G4Event* event );
  
  void EndOfEventAction   ( const G4Event* event );
  
  DsChromaEventAction  
  ( const std::string& type   ,
    const std::string& name   ,
    const IInterface*  parent ) ;
  
  virtual ~DsChromaEventAction( );

private:
  
  DsChromaEventAction();
  DsChromaEventAction( const DsChromaEventAction& );  
  DsChromaEventAction& operator=( const DsChromaEventAction& );

  void ChromaProcessing();

private:
  double  m_t0 ;
  Map_t   m_map ; 

 
};
#endif
