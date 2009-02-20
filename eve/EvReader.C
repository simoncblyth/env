/*

  IEvReader 
       abstract base class defining the interface that all Event Readers 
       must provide

  EvReader 
       factory class for IEvReader

*/


class IEvReader
{
  public:
     virtual void Read() = 0 ;
     virtual Bool_t LoadGeometry(const char* file, const char* shape  ) = 0 ;
     virtual Bool_t LoadProject( const char* file, const char* project) = 0 ;
     virtual Bool_t InitProject( const char* file, const char* project) = 0 ;

     virtual Bool_t LoadGeometry() = 0 ;
     virtual Bool_t InitProject() = 0 ;

     virtual ~IEvReader() ;
     ClassDef(IEvReader, 1 )     
};

ClassImp(IEvReader)
IEvReader::~IEvReader(){}



class AliEvReader ;


class EvReader
{
  public:
     enum ReaderType { kAlice , kAberdeen };
     static IEvReader* GetEvReader(ReaderType rt = kAlice );
  private:
    static IEvReader* gEvReader ; 
    static IEvReader* MakeEvReader(ReaderType rt);
};


IEvReader* EvReader::gEvReader = 0 ;
IEvReader* EvReader::GetEvReader(ReaderType rt)
{
    if( gEvReader == 0 ) gEvReader = MakeEvReader( rt );
    return gEvReader ;
}

IEvReader* EvReader::MakeEvReader(ReaderType rt)
{
       switch(rt)
       {
          case kAlice:
              return new AliEvReader ;
          default:
              return new AliEvReader ;
       }
}




