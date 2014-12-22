#ifndef G4DAEFOTON_H
#define G4DAEFOTON_H 


class G4DAEFoton {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

    enum {
       post_x,
       post_y,
       post_z,
       post_w,

       dirw_x,
       dirw_y,
       dirw_z,
       dirw_w,

       polw_x,
       polw_y,
       polw_z,
       polw_w,

       flag_x,
       flag_y,
       flag_z,
       flag_w
    };

};


#endif 


