#ifdef G4DAECHROMA_WITH_CPL

#include "G4DAEChroma/G4DAEChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEBuffer.hh"
#include "Chroma/ChromaPhotonList.hh"

#include "TMessage.h"
#include "ZMQRoot/MyTMessage.hh"

using namespace std ; 

const char* G4DAEChromaPhotonList::TMPL = "DAE_PATH_TEMPLATE_CPL" ;
const char* G4DAEChromaPhotonList::KEY  = "CPL" ;
const char* G4DAEChromaPhotonList::SHAPE  = "0,0" ; /* not used*/



G4DAEChromaPhotonList* G4DAEChromaPhotonList::Load(const char* evt, const char* key, const char* tmpl )
{
   ChromaPhotonList* cpl = ChromaPhotonList::Load(evt, key, tmpl);
   if(!cpl) return NULL ; 
   //cpl->Print();
   return new G4DAEChromaPhotonList(cpl);
}

G4DAEChromaPhotonList* G4DAEChromaPhotonList::LoadPath(const char* path, const char* key)
{
   ChromaPhotonList* cpl = ChromaPhotonList::LoadPath(path, key);
   if(!cpl) return NULL ; 
   //cpl->Print();
   return new G4DAEChromaPhotonList(cpl);
}


G4DAEChromaPhotonList* G4DAEChromaPhotonList::Slice( int a, int b )
{
    return new G4DAEChromaPhotonList( this, a, b );
}






G4DAEChromaPhotonList::G4DAEChromaPhotonList(ChromaPhotonList* cpl) : m_buffer(NULL), m_cpl(cpl)
{
}


G4DAEChromaPhotonList::G4DAEChromaPhotonList( G4DAEPhotons* src, int a, int b ) : m_buffer(NULL), m_cpl(NULL)
{
    m_cpl = new ChromaPhotonList ; 
    G4DAEPhotons::Transfer( this, src, a, b );
} 




G4DAEChromaPhotonList::G4DAEChromaPhotonList(std::size_t /*itemcapacity*/) : m_buffer(NULL), m_cpl(NULL)
{
   m_cpl = new ChromaPhotonList ; 
}
G4DAEChromaPhotonList::~G4DAEChromaPhotonList() 
{
   delete m_buffer ; 
   delete m_cpl ; 
}





// G4DAEPhotons
void G4DAEChromaPhotonList::AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid)
{
   m_cpl->AddPhoton(pos, mom, pol, _t, _wavelength, _pmtid );
}

void G4DAEChromaPhotonList::GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const {
   m_cpl->GetPhoton( index, pos, mom, pol, _t, _wavelength, _pmtid );
}
void G4DAEChromaPhotonList::Print(const char* msg) const 
{
   m_cpl->Print(msg);
}
void G4DAEChromaPhotonList::Details(bool hit) const 
{
   m_cpl->Details(hit);
}
std::size_t G4DAEChromaPhotonList::GetCount() const 
{
    return m_cpl->GetSize();
}
std::string G4DAEChromaPhotonList::GetDigest() const 
{
   return m_cpl->GetDigest();
}
void G4DAEChromaPhotonList::ClearAll() 
{
    return m_cpl->ClearAll();
}


// other
void G4DAEChromaPhotonList::Save(const char* evt, const char* key, const char* tmpl )
{
   m_cpl->Save(evt, key, tmpl);
}
void G4DAEChromaPhotonList::SavePath(const char* path, const char* key)
{
   m_cpl->SavePath(path, key );
}





// Serializable protocol methods

void G4DAEChromaPhotonList::SaveToBuffer()
{
   // serialize to buffer
   printf("G4DAEChromaPhotonList::SaveToBuffer \n");

   TMessage tmsg(kMESS_OBJECT);
   tmsg.WriteObject(m_cpl);

   delete m_buffer ; 
   m_buffer = new G4DAEBuffer(tmsg.Length(), tmsg.Buffer()); 

   printf("G4DAEChromaPhotonList::SaveToBuffer wrote %zu \n", m_buffer->GetSize() );
}

G4DAEChromaPhotonList* G4DAEChromaPhotonList::CreateOther( char* bytes, size_t size )
{
   MyTMessage* tmsg = new MyTMessage( reinterpret_cast<void*>(bytes), size );
   TObject* obj = tmsg->MyReadObject();
   G4DAEChromaPhotonList* other = new G4DAEChromaPhotonList((ChromaPhotonList*)obj) ;  
   // TODO: tmsg looks to leak, but had troubles previously 
   return other ; 
}


const char* G4DAEChromaPhotonList::GetBufferBytes()
{
   return m_buffer ? m_buffer->GetBytes() : NULL ;
}
std::size_t G4DAEChromaPhotonList::GetBufferSize()
{
   return m_buffer ? m_buffer->GetSize() : 0 ;
}
void G4DAEChromaPhotonList::DumpBuffer()
{
   printf("G4DAEChromaPhotonList::DumpBuffer \n");
   if(m_buffer) m_buffer->Dump() ;
}



#endif 
