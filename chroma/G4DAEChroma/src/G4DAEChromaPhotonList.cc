#include "G4DAEChroma/G4DAEChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEBuffer.hh"

#include "TMessage.h"
#include "ZMQRoot/MyTMessage.hh"

using namespace std ; 



G4DAEChromaPhotonList* G4DAEChromaPhotonList::Load(const char* evt, const char* key, const char* tmpl )
{
   ChromaPhotonList* cpl = ChromaPhotonList::Load(evt, key, tmpl);
   cpl->Print();
   return dynamic_cast<G4DAEChromaPhotonList*>(cpl);
   // hmm static passthroughs are tedious
}


  // itemcapacity not used, here to match G4DAEPhotonList 
G4DAEChromaPhotonList::G4DAEChromaPhotonList(std::size_t /*itemcapacity*/) : ChromaPhotonList(), m_buffer(NULL)
{
}
G4DAEChromaPhotonList::~G4DAEChromaPhotonList() 
{
   delete m_buffer ; 
}

// Serializable protocol methods

void G4DAEChromaPhotonList::SaveToBuffer()
{
   // serialize to buffer
   printf("G4DAEChromaPhotonList::SaveToBuffer \n");

   TMessage tmsg(kMESS_OBJECT);
   //tmsg.WriteObject((ChromaPhotonList*)this);
   tmsg.WriteObject(this);

   delete m_buffer ; 
   m_buffer = new G4DAEBuffer(tmsg.Length(), tmsg.Buffer()); 

   printf("G4DAEChromaPhotonList::SaveToBuffer wrote %zu \n", m_buffer->GetSize() );
}

G4DAEChromaPhotonList* G4DAEChromaPhotonList::Create( char* bytes, size_t size )
{
  // hmm difficult to do without class method due to nature of TObject deserialization
   MyTMessage* tmsg = new MyTMessage( reinterpret_cast<void*>(bytes), size );
   TObject* obj = tmsg->MyReadObject();
   return (G4DAEChromaPhotonList*)obj ;  
}


const char* G4DAEChromaPhotonList::GetBufferBytes()
{
   return m_buffer->GetBytes();
}
std::size_t G4DAEChromaPhotonList::GetBufferSize()
{
   return m_buffer->GetSize();
}
void G4DAEChromaPhotonList::DumpBuffer()
{
   printf("G4DAEChromaPhotonList::DumpBuffer \n");
   return m_buffer->Dump();
}



 
