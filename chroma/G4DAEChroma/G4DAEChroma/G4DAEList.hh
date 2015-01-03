#ifndef G4DAELIST_H 
#define G4DAELIST_H

#include "G4DAEChroma/G4DAEArrayHolder.hh"
#include "G4DAEChroma/G4DAEArray.hh"


template <class T> 
class G4DAEList : public G4DAEArrayHolder 
{
public:
  G4DAEList(G4DAEArrayHolder* holder);
  G4DAEList(G4DAEArrayHolder* holder, int start, int stop, int step);
  G4DAEList(G4DAEArray* array);
  G4DAEList(G4DAEArray* array, int start, int stop, int step);
  G4DAEList( std::size_t itemcapacity = 0, float* data = NULL);
  virtual ~G4DAEList();

// the below cannot go to base due to the static tmpl arguments
public:
  static std::string GetPath( const char* evt, const char* tmpl=T::TMPL);   
  static G4DAEList* Load(const char* evt, const char* key=T::KEY, const char* tmpl=T::TMPL);
  static G4DAEList* LoadPath(const char* path, const char* key=T::KEY);
  static G4DAEList* Adopt(G4DAEArrayHolder* holder);

public:
  virtual void Save(const char* evt, const char* key=T::KEY, const char* tmpl=T::TMPL );
  virtual void SavePath(const char* path, const char* key=T::KEY);

};

template <typename T>
G4DAEList<T>::G4DAEList( G4DAEArrayHolder* holder ) : G4DAEArrayHolder( holder ) 
{
}

template <typename T>
G4DAEList<T>::G4DAEList( G4DAEArrayHolder* holder, int start, int stop, int step ) : G4DAEArrayHolder( holder, start, stop, step ) 
{
}

template <typename T>
G4DAEList<T>::G4DAEList( G4DAEArray* array ) : G4DAEArrayHolder( array ) 
{
}

template <typename T>
G4DAEList<T>::G4DAEList( G4DAEArray* array, int start, int stop, int step ) : G4DAEArrayHolder( array, start, stop, step ) 
{
}

template <typename T>
G4DAEList<T>::G4DAEList( std::size_t itemcapacity, float* data) : G4DAEArrayHolder( itemcapacity, data, T::SHAPE ) 
{
}

template <typename T>
G4DAEList<T>::~G4DAEList() 
{
}

template <typename T>
void G4DAEList<T>::Save(const char* evt, const char* key, const char* tmpl )
{
    m_array->Save(evt, key, tmpl);
}

template <typename T>
void G4DAEList<T>::SavePath(const char* path, const char* key)
{
    m_array->SavePath(path, key);
}

template <typename T>
std::string G4DAEList<T>::GetPath( const char* evt, const char* tmpl)
{
    return G4DAEArray::GetPath(evt, tmpl);
}

template <typename T>
G4DAEList<T>* G4DAEList<T>::Load(const char* evt, const char* key, const char* tmpl)
{
    G4DAEArray* array = G4DAEArray::Load(evt, key, tmpl);
    return new G4DAEList<T>(array);
}

template <typename T>
G4DAEList<T>* G4DAEList<T>::LoadPath(const char* path, const char* key)
{
    G4DAEArray* array = G4DAEArray::LoadPath(path, key);
    return new G4DAEList<T>(array);
}




#endif
