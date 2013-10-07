# === func-gen- : geant4/clhep fgp geant4/clhep.bash fgn clhep fgh geant4
clhep-src(){      echo geant4/clhep.bash ; }
clhep-source(){   echo ${BASH_SOURCE:-$(env-home)/$(clhep-src)} ; }
clhep-vi(){       vi $(clhep-source) ; }
clhep-env(){      elocal- ; nuwa- ; }
clhep-usage(){ cat << EOU

CLHEP
=======


Rotation
---------

G4RotationMatrix::

    #include <CLHEP/Vector/Rotation.h>
    typedef CLHEP::HepRotation G4RotationMatrix;


external/build/LCG/clhep/2.0.4.2/CLHEP/Vector/Vector/Rotation.h::

    178   // ------------  axis & angle of rotation:
    179   inline  double  getDelta() const;
    180   inline  Hep3Vector getAxis () const;
    181   double     delta() const;
    182   Hep3Vector    axis () const;
    183   HepAxisAngle  axisAngle() const;
    184   void getAngleAxis(double & delta, Hep3Vector & axis) const;
    185   // Returns the rotation angle and rotation axis (Geant4).     [Rotation.cc]


global/HEPGeometry/include/G4Transform3D.hh::

     33 #include "globals.hh"
     34 #include <CLHEP/Geometry/Transform3D.h>
     35 
     36 typedef HepGeom::Transform3D G4Transform3D;
     37 
     38 typedef HepGeom::Rotate3D G4Rotate3D;
     39 typedef HepGeom::RotateX3D G4RotateX3D;
     40 typedef HepGeom::RotateY3D G4RotateY3D;

external/build/LCG/clhep/2.0.4.2/CLHEP/Geometry/Geometry/Transform3D.h::

    /// Rotate3D classes provides various ways to construct the transformation matrix
    ///  
    ///
    172   class Transform3D {
    173   protected:
    174     double xx_, xy_, xz_, dx_,     // 4x3  Transformation Matrix
    175            yx_, yy_, yz_, dy_,
    176            zx_, zy_, zz_, dz_;
    177 
    178     // Protected constructor
    179     Transform3D(double XX, double XY, double XZ, double DX,
    180         double YX, double YY, double YZ, double DY,
    181         double ZX, double ZY, double ZZ, double DZ)
    182       : xx_(XX), xy_(XY), xz_(XZ), dx_(DX),
    183     yx_(YX), yy_(YY), yz_(YZ), dy_(DY),
    184     zx_(ZX), zy_(ZY), zz_(ZZ), dz_(DZ) {}





EOU
}
clhep-dir(){ echo $(nuwa-clhep-bdir) ; }
clhep-cd(){  cd $(clhep-dir); }
