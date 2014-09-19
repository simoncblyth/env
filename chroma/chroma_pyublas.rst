Chroma Use of PyUblas
=======================

Chroma uses a Python extension that via pyublas/boost-python/numpy magic
succeeds to pull numpy arrays out of a C++ class member objects
such as  `std::vector<G4ThreeVector> pos`

The `pyublas::numpy_vector<double>` templated type is **cool**, 
it is usable from C++ with `std::` techniques and yet 
also can be passed to python as a numpy array

* http://documen.tician.de/pyublas/cpptypes.html


using extension from ipython
-------------------------------

::

    In [6]: import chroma.generator._g4chroma as _g4chroma

    In [7]: _g4chroma.
    _g4chroma.ChromaPhysicsList     _g4chroma.PhotonTrackingAction  

    In [7]: _g4chroma.PhotonTrackingAction.GetDirX?
    Type:       instancemethod
    String Form:<unbound method PhotonTrackingAction.GetDirX>
    Docstring:
    GetDirX( (PhotonTrackingAction)arg1) -> object :

        C++ signature :
            pyublas::numpy_vector<double> GetDirX(PhotonTrackingAction const*)

    In [8]: pta = _g4chroma.PhotonTrackingAction()
    *** G4Exception : Tracking0001
          issued by : G4UserTrackingAction::G4UserTrackingAction()
     You are instantiating G4UserTrackingAction BEFORE your
    G4VUserPhysicsList is instantiated and assigned to G4RunManager.
     Such an instantiation is prohibited since Geant4 version 8.0. To fix this problem,
    please make sure that your main() instantiates G4VUserPhysicsList AND
    set it to G4RunManager before instantiating other user action classes
    such as G4UserTrackingAction.

    ---------------------------------------------------------------------------
    AssertionError                            Traceback (most recent call last)
    AssertionError: *** Fatal Exception ***
    Severity : 


    In [9]: import Geant4

    In [10]: import g4py

    In [11]: g4py.__file__
    Out[11]: '/usr/local/env/chroma_env/lib/python2.7/site-packages/g4py/__init__.pyc'

    In [12]: g4py??
    Type:       module
    String Form:<module 'g4py' from '/usr/local/env/chroma_env/lib/python2.7/site-packages/g4py/__init__.pyc'>
    File:       /usr/local/env/chroma_env/lib/python2.7/site-packages/g4py/__init__.py
    Source:
    # $Id: __init__.py,v 1.1 2008-12-01 07:02:47 kmura Exp $
    # Geant4Py site-modules

    In [14]: Geant4.gRunManager
    Out[14]: <Geant4.G4run.G4RunManager at 0x10a591e68>

    In [15]: physics_list = _g4chroma.ChromaPhysicsList()

    In [17]: Geant4.gRunManager.SetUserInitialization(physics_list)

    In [18]: tracking_action = _g4chroma.PhotonTrackingAction()

    In [19]: tracking_action.GetNumPhotons()
    Out[19]: 0


test setters
-------------

::

    In [1]: import chroma.generator._g4chroma as _g4chroma

    In [2]: _g4chroma.PhotonTrackingAction.SetX?
    Type:       instancemethod
    String Form:<unbound method PhotonTrackingAction.SetX>
    Docstring:
    SetX( (PhotonTrackingAction)arg1, (numpy.ndarray)arg2) -> None :

        C++ signature :
            void SetX(PhotonTrackingAction*,pyublas::numpy_vector<double>)

    In [3]: from Geant4 import gRunManager

    In [4]: physics_list = _g4chroma.ChromaPhysicsList()

    In [5]: gRunManager.SetUserInitialization(physics_list)

    In [6]: pta = _g4chroma.PhotonTrackingAction()

    In [7]: pta.GetNumPhotons()
    Out[7]: 0

    In [8]: pta.AssignNumPhotons(1000)

    In [9]: pta.GetNumPhotons()
    Out[9]: 1000


    In [11]: import numpy as np

    In [27]: x = 10*np.ones(1000,np.float32)

    In [28]: pta.SetX(x)
    ---------------------------------------------------------------------------
    ArgumentError                             Traceback (most recent call last)
    <ipython-input-28-d38581056cf9> in <module>()
    ----> 1 pta.SetX(x)

    ArgumentError: Python argument types in
        PhotonTrackingAction.SetX(PhotonTrackingAction, numpy.ndarray)
    did not match C++ signature:
        SetX(PhotonTrackingAction*, pyublas::numpy_vector<double>)

    In [29]: x = 10*np.ones(1000,np.float64)

    In [33]: x[10] = 123

    In [34]: pta.SetX(x)

    In [38]: xx2 = pta.GetX()

    In [39]: xx2
    Out[39]: 
    array([  10.,   10.,   10.,   10.,   10.,   10.,   10.,   10.,   10.,
             10.,  123.,   10.,   10.,   10.,   10.,   10.,   10.,   10.,




extension setup
-----------------

`chroma/setup.py`::

     //   config to find libs and headers skipped
     //
     59 setup(
     60     name = 'Chroma',
     61     version = '0.5',
     62     packages = find_packages(),
     63     include_package_data=True,
     64 
     65     scripts = ['bin/chroma-sim', 'bin/chroma-cam',
     66                'bin/chroma-geo', 'bin/chroma-bvh',
     67                'bin/chroma-server'],
     68     ext_modules = [
     69         Extension('chroma.generator._g4chroma',
     70                   ['src/G4chroma.cc'],
     71                   include_dirs=include_dirs,
     72                   extra_compile_args=geant4_cflags,
     73                   extra_link_args=geant4_libs+clhep_libs,
     74                   extra_objects=extra_objects,
     75                   libraries=libraries,
     76                   ),
     77         Extension('chroma.generator.mute',
     78                   ['src/mute.cc'],
     79                   include_dirs=include_dirs,
     80                   extra_compile_args=geant4_cflags,
     81                   extra_link_args=geant4_libs+clhep_libs,
     82                   extra_objects=extra_objects,
     83                   libraries=libraries),
     84         ],
     85 
     86     setup_requires = ['pyublas'],


build extension with setters added
------------------------------------

::

    (chroma_env)delta:chroma blyth$ python setup.py build_ext
    running build_ext
    building 'chroma.generator._g4chroma' extension
    C compiler: /usr/bin/clang -fno-strict-aliasing -fno-common -dynamic -pipe -Os -fwrapv -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes

    compile options: '-Isrc -I/usr/local/env/chroma_env/lib/python2.7/site-packages/pyublas/include -I/usr/local/env/chroma_env/include -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -c'
    extra options: '-DG4INTY_USE_XT -I/usr/X11R6/include -I/opt/local/include -DG4UI_USE_TCSH -DG4VIS_USE_RAYTRACERX -I/usr/local/env/chroma_env/bin/../include/Geant4'
    clang: src/G4chroma.cc
    In file included from src/G4chroma.cc:136:
    In file included from /usr/local/env/chroma_env/lib/python2.7/site-packages/pyublas/include/pyublas/numpy.hpp:40:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h:4:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarrayobject.h:17:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarraytypes.h:1760:
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: "Using deprecated NumPy API, disable it by "          "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
    #warning "Using deprecated NumPy API, disable it by " \
     ^
    1 warning generated.
    /usr/bin/clang++ -bundle -undefined dynamic_lookup -L/opt/local/lib -Wl,-headerpad_max_install_names -L/opt/local/lib/db46 build/temp.macosx-10.9-x86_64-2.7/src/G4chroma.o -lboost_python -o build/lib.macosx-10.9-x86_64-2.7/chroma/generator/_g4chroma.so -L/usr/local/env/chroma_env/bin/../lib -lG4Tree -lG4FR -lG4GMocren -lG4visHepRep -lG4RayTracer -lG4VRML -lG4vis_management -lG4modeling -lG4interfaces -lG4persistency -lG4analysis -lG4error_propagation -lG4readout -lG4physicslists -lG4run -lG4event -lG4tracking -lG4parmodels -lG4processes -lG4digits_hits -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms -lG4global -lG4clhep
    (chroma_env)delta:chroma blyth$ 





Usage of G4chroma extension
------------------------------

`chroma/chroma/generator/g4gen.py`::

    003 import pyublas
     ..
     14 from chroma.generator import _g4chroma
     .. 
     19 class G4Generator(object):
     20     def __init__(self, material, seed=None):
     ..  
     50         self.tracking_action = _g4chroma.PhotonTrackingAction()
     51         gRunManager.SetUserAction(self.tracking_action)
     ..
     79     def _extract_photons_from_tracking_action(self, sort=True):
     80         n = self.tracking_action.GetNumPhotons()
     81         pos = np.zeros(shape=(n,3), dtype=np.float32)
     82         pos[:,0] = self.tracking_action.GetX()
     83         pos[:,1] = self.tracking_action.GetY()
     84         pos[:,2] = self.tracking_action.GetZ()
     .. 
     96         wavelengths = self.tracking_action.GetWavelength().astype(np.float32)
     97 
     98         t0 = self.tracking_action.GetT0().astype(np.float32)
     ..
    108         return Photons(pos, dir, pol, wavelengths, t0)


G4chroma extension
---------------------

`chroma/src/G4chroma.hh`::

     17 class PhotonTrackingAction : public G4UserTrackingAction
     18 {
     19 public:
     20   PhotonTrackingAction();
     21   virtual ~PhotonTrackingAction();
     22 
     23   int GetNumPhotons() const;
     24   void Clear();
     25 
     26   void GetX(double *x) const;
     27   void GetY(double *y) const;
     28   void GetZ(double *z) const;
     29   void GetDirX(double *dir_x) const;
     30   void GetDirY(double *dir_y) const;
     31   void GetDirZ(double *dir_z) const;
     32   void GetPolX(double *pol_x) const;
     33   void GetPolY(double *pol_y) const;
     34   void GetPolZ(double *pol_z) const;
     35 
     36   void GetWavelength(double *wl) const;
     37   void GetT0(double *t) const;
     38 
     39   virtual void PreUserTrackingAction(const G4Track *);
     40 
     41 protected:
     42   std::vector<G4ThreeVector> pos;
     43   std::vector<G4ThreeVector> dir;
     44   std::vector<G4ThreeVector> pol;
     45   std::vector<double> wavelength;
     46   std::vector<double> t0;
     47 };




`chroma/src/G4chroma.cc`::

    ///
    /// fills array passed from contained vector
    ///
    050 void PhotonTrackingAction::GetX(double *x) const
     51 {
     52   for (unsigned i=0; i < pos.size(); i++) x[i] = pos[i].x();
     53 }
    ...
    118 #include <boost/python.hpp>
    119 #include <pyublas/numpy.hpp>
    120 
    121 using namespace boost::python;
    122 
    ///
    ///   becomes the `pta.GetX()` python member func returning numpy arrays
    ///
    123 pyublas::numpy_vector<double> PTA_GetX(const PhotonTrackingAction *pta)
    124 {
    125   pyublas::numpy_vector<double> r(pta->GetNumPhotons());
    126   pta->GetX(&r[0]);
    127   return r;
    128 }
    ...
    201 void export_Chroma()
    202 {
    203   class_<ChromaPhysicsList, ChromaPhysicsList*, bases<G4VModularPhysicsList>, boost::noncopyable > ("ChromaPhysicsList", "EM+Optics physics list")
    204     .def(init<>())
    205     ;
    206 
    207   class_<PhotonTrackingAction, PhotonTrackingAction*, bases<G4UserTrackingAction>,
    208      boost::noncopyable > ("PhotonTrackingAction", "Tracking action that saves photons")
    209     .def(init<>())
    210     .def("GetNumPhotons", &PhotonTrackingAction::GetNumPhotons)
    211     .def("Clear", &PhotonTrackingAction::Clear)
    212     .def("GetX", PTA_GetX)
    213     .def("GetY", PTA_GetY)
    214     .def("GetZ", PTA_GetZ)
    215     .def("GetDirX", PTA_GetDirX)
    216     .def("GetDirY", PTA_GetDirY)
    217     .def("GetDirZ", PTA_GetDirZ)
    218     .def("GetPolX", PTA_GetPolX)
    219     .def("GetPolY", PTA_GetPolY)
    220     .def("GetPolZ", PTA_GetPolZ)
    221     .def("GetWavelength", PTA_GetWave)
    222     .def("GetT0", PTA_GetT0)
    223     ;
    224 }
    225 
    226 BOOST_PYTHON_MODULE(_g4chroma)
    227 {
    228   export_Chroma();
    229 }



PyUblas
--------

* http://mathema.tician.de/software/pyublas/

PyUblas provides a seamless glue layer between Numpy and Boost.Ublas for use with Boost.Python.

* http://documen.tician.de/pyublas/
* http://www.boost.org/libs/numeric/ublas


* http://documen.tician.de/pyublas/cpptypes.html










