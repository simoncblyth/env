#include "G4DAEUtil.hh"
//#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>


int G4DAEUtil::testNCNameDemo()
{
   std::string id = "/dd/Geometry/Pool/lvNearPoolOWS#pvVetoPmtNearOutFacein#pvNearOutFaceinWall7#pvNearOutFaceinWall7:15#pvVetoPmtUnit#pvPmtMount#pvMountRib2s#pvMountRib2s:1#pvMountRib2unit0xc3f0d18_pos" ;
   return testNCName(id);
}


int G4DAEUtil::testNCName(std::string& id )
{
   std::string orig(id);
   std::cout << "[orig]    " << id << std::endl ;

   int rc ;
   rc = encodeNCName(id);
   std::cout << "[encode] " << rc << " " << id << std::endl ;
   rc = decodeNCName(id);
   std::cout << "[decode] " << rc << " " << id << std::endl ;

   if( id != orig )
   {
      std::cout << "FAILED TO ROUNDTRIP " << std::endl ;
      std::cout << "[orig]   " << orig << std::endl ;
      std::cout << "[id]     " << id << std::endl ;
   }

   return 0 ; 
}





void G4DAEUtil::replaceAll(std::string& id, std::string const& from, std::string const& to)
{
    std::size_t lookHere = 0;
    std::size_t foundHere;
    while((foundHere = id.find(from, lookHere)) != std::string::npos)
    {
          id.replace(foundHere, from.size(), to);
          lookHere = foundHere + to.size();
    }
}

int G4DAEUtil::transformNCName( std::string& id, bool encode )
{

   int ierr = 0 ;
   typedef std::map<std::string, std::string> Map ;

   Map m; 
   m["/"] = "__" ; 
   m[":"] = ".." ; 
   m["#"] = "--" ; 

   Map::iterator i ;

   for(i=m.begin(); i!=m.end() ; ++i )
   { 
       std::size_t pos = id.find( i->second ) ;
       if( pos != std::string::npos && encode ){
           std::cout << "WARNING transformNCName while encoding finds output tokens " << i->second << " at " << pos << std::endl ;
           ierr += 1 ; 
       }
   }

   for(i=m.begin(); i!=m.end() ; ++i )
   {
       if(encode)
       {
            //std::cout << i->first << " => " << i->second << " (encode)" << std::endl ; 
            replaceAll(id, i->first, i->second );
       }
       else
       {
            //std::cout << i->second << " => " << i->first << " (decode)" << std::endl ; 
            replaceAll(id, i->second, i->first );
       }     
   } 

   return ierr ; 
} 
int G4DAEUtil::decodeNCName( std::string& id )
{
    return transformNCName( id, false );   
}
int G4DAEUtil::encodeNCName( std::string& id )
{
    return transformNCName( id, true );   
}

void G4DAEUtil::WriteLines(std::string& fname, std::vector<std::string>& lines)
{
   std::cout << "G4DAEUtil::WriteLines " << fname << std::endl ;
   std::fstream fs;
   fs.open ( fname.c_str(), std::fstream::out );
   std::vector<std::string>::iterator it ;
   for( it = lines.begin() ; it != lines.end() ; it++ ) fs << *it << std::endl ;
   fs.close();
}


