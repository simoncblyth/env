#ifndef G4DAEFOTON_H
#define G4DAEFOTON_H 

class G4DAEFoton  {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

    enum {
       _post_x,
       _post_y,
       _post_z,
       _post_w,

       _dirw_x,
       _dirw_y,
       _dirw_z,
       _dirw_w,

       _polw_x,
       _polw_y,
       _polw_z,
       _polw_w,

       _flag_x,
       _flag_y,
       _flag_z,
       _flag_w
    };


};


#endif 


