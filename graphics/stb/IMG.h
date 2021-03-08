#pragma once

#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cassert>

struct IMG 
{
   int width ; 
   int height ; 
   int channels ; 
   const unsigned char* data ; 
   const char* loadpath ; 
   const char* loadext ; 

   IMG(const char* path); 
   IMG(int width_, int height_, int channels_, const unsigned char* data_); 
   virtual ~IMG();  

   std::string desc() const ; 

   void writePNG() const ; 
   void writeJPG(int quality) const ; 

   void writePNG(const char* path) const ; 
   void writeJPG(const char* path, int quality) const ; 

   void writePNG(const char* dir, const char* name) const ; 
   void writeJPG(const char* dir, const char* name, int quality) const ; 

   static std::string FormPath(const char* dir, const char* name);
   static bool EndsWith( const char* s, const char* q);
   static const char* ChangeExt( const char* s, const char* x1, const char* x2);
   static const char* Ext(const char* path);
};

#include "IMG.h"

#ifdef IMG_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image.h"
#include "stb_image_write.h"


inline bool IMG::EndsWith( const char* s, const char* q) // static 
{
    int pos = strlen(s) - strlen(q) ;
    return pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 ; 
}

inline const char* IMG::ChangeExt( const char* s, const char* x1, const char* x2)  // static
{
    assert( EndsWith(s, x1) );  

    std::string st = s ; 
    std::stringstream ss ; 

    ss << st.substr(0, strlen(s) - strlen(x1) ) ; 
    ss << x2 ;   
    std::string ns = ss.str() ; 
    return strdup(ns.c_str()); 
}


inline IMG::IMG(const char* path) 
    :
    width(0),
    height(0),
    channels(0),
    data(stbi_load(path, &width, &height, &channels, 0)),
    loadpath(strdup(path)),
    loadext(Ext(loadpath))
{
}

inline IMG::IMG(int width_, int height_, int channels_, const unsigned char* data_) 
    :
    width(width_),
    height(height_),
    channels(channels_),
    data(data_),
    loadpath("image.ppm"),
    loadext(Ext(loadpath))
{
}

inline IMG::~IMG()
{
    stbi_image_free((void*)data);    
}

inline std::string IMG::desc() const
{
    std::stringstream ss ;
    ss << "IMG"
       << " width " << width 
       << " height " << height
       << " channels " << channels
       << " loadpath " << ( loadpath ? loadpath : "-" )
       << " loadext " << ( loadext ? loadext : "-" )
       ;
    std::string s = ss.str(); 
    return s ;
}

inline std::string IMG::FormPath(const char* dir, const char* name)
{
    std::stringstream ss ;
    ss << dir << "/" << name ; 
    std::string s = ss.str(); 
    return s ; 
}

inline void IMG::writePNG() const 
{
    assert(loadpath && loadext); 
    const char* pngpath = ChangeExt(loadpath, loadext, ".png" ); 
    if(strcmp(pngpath, loadpath)==0)
    {
        std::cout << "IMG::writePNG ERROR cannot overwrite loadpath " << loadpath << std::endl ; 
    } 
    else
    {
         writePNG(pngpath);  
    }
}

inline void IMG::writeJPG(int quality) const 
{
    assert(loadpath && loadext); 

    std::stringstream ss ; 
    ss << "_" << quality << ".jpg" ; 
    std::string x = ss.str(); 

    const char* jpgpath = ChangeExt(loadpath, loadext, x.c_str() ); 

    if(strcmp(jpgpath, loadpath)==0)
    {
        std::cout << "IMG::writeJPG ERROR cannot overwrite loadpath " << loadpath << std::endl ; 
    } 
    else
    {
        writeJPG(jpgpath, quality);  
    }
}


inline void IMG::writePNG(const char* dir, const char* name) const 
{
    std::string s = FormPath(dir, name); 
    writePNG(s.c_str()); 
}
inline void IMG::writePNG(const char* path) const 
{
    std::cout << "stbi_write_png " << path << std::endl ; 
    stbi_write_png(path, width, height, channels, data, width * channels);
}


inline void IMG::writeJPG(const char* dir, const char* name, int quality) const 
{
    std::string s = FormPath(dir, name); 
    writeJPG(s.c_str(), quality); 
}
inline void IMG::writeJPG(const char* path, int quality) const 
{
    std::cout << "stbi_write_jpg " << path << " quality " << quality << std::endl ; 
    assert( quality > 0 && quality <= 100 ); 
    stbi_write_jpg(path, width, height, channels, data, quality );
}

inline const char* IMG::Ext(const char* path)
{
    std::string s = path ; 
    std::size_t pos = s.find_last_of(".");
    std::string ext = s.substr(pos) ;  
    return strdup(ext.c_str()); 
}



