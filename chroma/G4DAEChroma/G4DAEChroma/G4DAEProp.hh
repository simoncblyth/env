#ifndef G4DAEPROP_H
#define G4DAEPROP_H 

class G4DAEArrayHolder ; 
class G4PhysicsOrderedFreeVector ;

class G4DAEProp {
    public:

    static const char* TMPL ;   // name of envvar containing path template 
    static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
    static const char* KEY ;  

    static G4DAEArrayHolder* Copy(G4PhysicsOrderedFreeVector* pofv, double xscale=1.0, double yscale=1.0 );
    static G4PhysicsOrderedFreeVector* CreatePOFV(G4DAEArrayHolder* holder, double xscale=1.0, double yscale=1.0);

    enum {
       _binEdge,      //  0
       _binValue,
       SIZE

    };

};


#endif 


