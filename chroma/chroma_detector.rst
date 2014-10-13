Chroma Modeling of Sensitive Detectors 
=========================================

.. contents:: :local:

Overview
--------

#. Chroma/DetSim mismatch: 

   * DetSim, QE handled as a Bialkali material EFFICIENCY property
   * Chroma, **Surface.detect** property 

#. To raise SURFACE_DETECT in Chroma need to make PMT Cathods 
   into surfaces with **Surface.detect** property corresponding to 
   the QE (matching what ProcessHits does) 



GPU hit formation
-------------------

cutdown CPL to just the hits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Non-trivial, with GPU processing to make such selections.
Study chroma slicing.

ChromaManager singleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* develop as part of pkg NuWa-trunk/dybgaudi/Utilities/Chroma
  (probably rename to G4DAEChroma)

* where to construct the transform cache ?  need access from within 
  a StackAction (or other G4 level class) so a G4 singleton manager
  seems like the most natural

Runtime interface to Chroma GPU propagation, send photons get hits in reply
(or at least hit data with which to form them). 

Grab everything possible from the StackAction into *ChromaManager* to 
avoid duplication.
Initially considered naming *G4DAEChroma* 
but that could be confusing as this will clearly 
never be able to be sent upstream like the *G4DAE* exporter.

* As considering to move photon collection into the processes, to avoid stack addition/deletion overhead.
* also it would be cleaner and more reusable to keep the **simulation** aspect 
  distinct from **analysis** aspects

Responsible for

* batching configuration
* photon collection, 
* serialization, sending to GPU, 
* getting reply, deserialization, forming hits 
* holding the transform cache

Holding transform cache avoids transporting 
both global and local coordinates for pos/pol/mon
in the transport class for millions of photons (or thousands? of hits).

* will need a volume index, as well as channel_id


G4VSensitiveDetector
~~~~~~~~~~~~~~~~~~~~~~~

I'm killing all generated OP from G4 point, does that allow
me other optimizations.  

* How can I hook into the standard hit collection machinery ? G4HCOfThisEvent

* http://www-geant4.kek.jp/g4users/g4tut07/docs/SensitiveDetector.pdf



correspondence between the COLLADA `boundgeom.matrix` and G4 TopTransform
----------------------------------------------------------------------------

For forming Hits on GPU need to do local coordinate transform
somewhere as ProcessHits provides the local coordinate of hits wrt to the
sensitive volume (PMT Cathode).

Added transform dumping to gausstools `GiGaRunActionExport::WriteIdMap`
and compare with boundgeom matrices of nodes.

volume 0
~~~~~~~~~~

/data1/env/local/env/geant4/geometry/export/DayaBay_MX_20141013-1542/g4_00.idmap::

   .1 # GiGaRunActionExport::WriteIdMap fields: index,pmtid,pmtid(hex),pvname  npv:12230
    2 0 0 0  (0,0,0)
    3    [ (           1             0             0)
    4      (           0             1             0)
    5      (           0             0             1) ]
    6  Universe


volume 1
~~~~~~~~~~

::

   .7 1 0 0  (664494,-449556,2110)
    8    [ (   -0.543174      -0.83962             0)
    9      (     0.83962     -0.543174             0)
   10      (           0             0             1) ]
   11  /dd/Structure/Sites/db-rock


::

    In [13]: np.set_printoptions(precision=5, suppress=True)

    In [27]: m1 = DAENode.get("1").boundgeom.matrix
    2014-10-13 16:11:40,609 env.geant4.geometry.collada.g4daenode:686 INFO     arg 1 => indices [1] => node   __dd__Structure__Sites__db-rock0xc15d358.0             __dd__Materials__Rock0xc0300c8  

    In [35]: m1
    Out[35]: 
    array([[     -0.54317,      -0.83962,       0.     ,  -16520.     ],
           [      0.83962,      -0.54317,       0.     , -802110.     ],
           [      0.     ,       0.     ,       1.     ,   -2110.     ],
           [      0.     ,       0.     ,       0.     ,       1.     ]], dtype=float


    In [28]: invert_homogenous(m1)
    Out[28]: 
    array([[     -0.54317,       0.83962,       0.     ,  664494.35857],
           [     -0.83962,      -0.54317,       0.     , -449555.84222],
           [      0.     ,       0.     ,       1.     ,    2110.     ],
           [      0.     ,       0.     ,       0.     ,       1.     ]])


* some correspondence but there is a definition difference to clear up


last volume
~~~~~~~~~~~~~~

::

    61147 12229 0 0  (664494,-449556,12410)
    61148    [ (   -0.543174      -0.83962             0)
    61149      (     0.83962     -0.543174             0)
    61150      (           0             0             1) ]
    61151  /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab9


    In [16]: m = DAENode.get("12229").boundgeom.matrix
    2014-10-13 15:57:01,308 env.geant4.geometry.collada.g4daenode:686 INFO     arg 12229 => indices [12229] => node   __dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab90xc15cf08.0             __dd__Materials__RadRock0xcd2f508  

    In [17]: m
    Out[17]: 
    array([[     -0.54317,      -0.83962,       0.     ,  -16520.     ],
           [      0.83962,      -0.54317,       0.     , -802110.     ],
           [      0.     ,       0.     ,       1.     ,  -12410.     ],
           [      0.     ,       0.     ,       0.     ,       1.     ]], dtype=float32)

    In [25]: from env.geant4.geometry.collada.g4daeview.daeutil import invert_homogenous

    In [26]: invert_homogenous( m )    # translation result matches, rotation is transposed 
    Out[26]: 
    array([[     -0.54317,       0.83962,       0.     ,  664494.35857],
           [     -0.83962,      -0.54317,       0.     , -449555.84222],
           [      0.     ,       0.     ,       1.     ,   12410.     ],
           [      0.     ,       0.     ,       0.     ,       1.     ]])



all volumes compared
~~~~~~~~~~~~~~~~~~~~~~

Loosening tolerances succeeds to get all volumes to match.  

* `env/geant4/geometry/collada/check_volume_transforms.py`



geant4 transforms
~~~~~~~~~~~~~~~~~~~

`source/geometry/management/include/G4AffineTransform.hh`::

    .69 class G4AffineTransform
     70 {
     71 
     72 public:
     73 
     74   G4AffineTransform();
     75 
     76 public: // with description
     77 
     78   G4AffineTransform(const G4ThreeVector &tlate);
     79     // Translation only: under t'form translate point at origin by tlate
     80 
     81   G4AffineTransform(const G4RotationMatrix &rot);
     82     // Rotation only: under t'form rotate by rot
     83 
     84   G4AffineTransform(const G4RotationMatrix &rot,
     85                     const G4ThreeVector &tlate);
     86     // Under t'form: rotate by rot then translate by tlate
     87 
     88   G4AffineTransform(const G4RotationMatrix *rot,
     89                     const G4ThreeVector &tlate);
     90     // Optionally rotate by *rot then translate by tlate - rot may be null
     ..     
    113   G4ThreeVector TransformPoint(const G4ThreeVector &vec) const;
    114     // Transform the specified point: returns vec*rot+tlate
    115 
    116   G4ThreeVector TransformAxis(const G4ThreeVector &axis) const;
    117     // Transform the specified axis: returns
    118 
    119   void ApplyPointTransform(G4ThreeVector &vec) const;
    120     // Transform the specified point (in place): sets vec=vec*rot+tlate
    121 
    122   void ApplyAxisTransform(G4ThreeVector &axis) const;
    123     // Transform the specified axis (in place): sets axis=axis*rot;



pycollada boundgeom nodes matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binding is done by `g4daenode.py` based on the **full** geometry of the COLLADA export.
Partial exports at C++ level would result in binding (ie coordinate system)
based on a different **Universe**. 

Every bound node has a 4x4 matrix associated::

    delta:~ blyth$ g4daenode.sh -i 
    2014-10-13 14:42:13,194 env.geant4.geometry.collada.g4daenode:2344 INFO     /Users/blyth/env/bin/g4daenode.py
    2014-10-13 14:42:13,194 env.geant4.geometry.collada.g4daenode:2289 INFO     Using pvar DAE_NAME_DYB to resolve path : /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae 
    2014-10-13 14:42:13,261 env.geant4.geometry.collada.idmap:165 INFO     found 685 unique ids 
    ...

    In [1]: DAENode.get("0").boundgeom.matrix
    2014-10-13 14:42:28,560 env.geant4.geometry.collada.g4daenode:686 INFO     arg 0 => indices [0] => node   top.0             __dd__Materials__Vacuum0xbf9fcc0  
    Out[1]: 
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]], dtype=float32)


    In [3]: DAENode.get("1000").boundgeom.matrix
    2014-10-13 14:42:39,694 env.geant4.geometry.collada.g4daenode:686 INFO     arg 1000 => indices [1000] => node   __dd__Geometry__RPC__lvRPCGasgap23--pvStrip23Array--pvStrip23ArrayOne..6--pvStrip23Unit0xc128768.46             __dd__Materials__MixGas0xc21d930  
    Out[3]: 
    array([[ -5.39289117e-01,  -8.41108799e-01,  -4.12638858e-02,
             -1.99534609e+04],
           [  8.42120171e-01,  -5.38641453e-01,  -2.64251661e-02,
             -7.99555000e+05],
           [  0.00000000e+00,  -4.89999987e-02,   9.98799026e-01,
             -1.36956641e+03],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              1.00000000e+00]], dtype=float32)



Raw unbound COLLADA matrix elements are written by `G4DAEWriteStructure::TraverseVolumeTree`::

      G4Transform3D daughterR;

      daughterR = TraverseVolumeTree(physvol->GetLogicalVolume(),depth+1);

      G4RotationMatrix rot, invrot;
      if (physvol->GetFrameRotation() != 0)
      {
         rot = *(physvol->GetFrameRotation());
         invrot = rot.inverse();
      }

      // G4Transform3D P(rot,physvol->GetObjectTranslation());  GDML does this : not inverting the rotation portion 
      G4Transform3D P(invrot,physvol->GetObjectTranslation());

      PhysvolWrite(nodeElement,physvol,invR*P*daughterR,ModuleName);


The unbound matrices just provide the position of volume relative to parent.  When 
binding is done by the **objects** call in  `g4daenode.py`::

     564         dae = collada.Collada(path)
     565         log.debug("pycollada parse completed ")
     566         boundgeom = list(dae.scene.objects('geometry'))
     567         top = dae.scene.nodes[0]
     568         log.debug("pycollada binding completed, found %s  " % len(boundgeom))

the node tree heirarchy of matrices are multiplied to arrive at the 
scene graph bound matrix. 
  
ProcessHits Hit formation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.cc`::

    333     const G4TouchableHistory* hist =
    334         dynamic_cast<const G4TouchableHistory*>(preStepPoint->GetTouchable());
    ...
    340     const DetectorElement* de = this->SensDetElem(*hist);
    341     if (!de) return false;
    342 
    343     // wangzhe QE calculation starts here.
    344     int pmtid = this->SensDetId(*de);
    ...
    459     DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
    460 
    461     // base hit
    462 
    463     // Time since event created
    464     sphit->setHitTime(preStepPoint->GetGlobalTime());
    465 
    466     //#include "G4NavigationHistory.hh"
    467 
    468     const G4AffineTransform& trans = hist->GetHistory()->GetTopTransform();
    469     const G4ThreeVector& global_pos = preStepPoint->GetPosition();
    470     G4ThreeVector pos = trans.TransformPoint(global_pos);
    471     sphit->setLocalPos(pos);
    472     sphit->setSensDetId(pmtid);
    473    
    474     // pmt hit
    475     // sphit->setDir(...);       // for now
    476     G4ThreeVector pol = trans.TransformAxis(track->GetPolarization());
    477     pol = pol.unit();
    478     G4ThreeVector dir = trans.TransformAxis(track->GetMomentum());
    479     dir = dir.unit();
    480     sphit->setPol(pol);
    481     sphit->setDir(dir);
    482     sphit->setWavelength(wavelength);
    483     sphit->setType(0);
    484     // G4cerr<<"PMT: set hit weight "<<weight<<G4endl; //gonchar
    485     sphit->setWeight(weight);


::

    delta:geant4.10.00.p01 blyth$ find source -name '*.hh'  -exec grep -H GetTopTransform {} \;
    source/geometry/volumes/include/G4NavigationHistory.hh:  inline const G4AffineTransform& GetTopTransform() const; 




    90   inline const G4AffineTransform& GetTopTransform() const;
    91     // Returns topmost transform.

    145 #if defined(WIN32)
    146   std::vector<G4NavigationLevel> fNavHistory;
    147 #else
    148   std::vector<G4NavigationLevel,
    149               G4EnhancedVecAllocator<G4NavigationLevel> > fNavHistory;
    150     // The geometrical tree; uses specialized allocator to optimize memory
    151     // handling, reduce possible fragmentation and use of malloc in MT mode
    152 #endif
    153 
    154   G4int fStackDepth;
    155     // Depth of stack: effectively depth in geometrical tree


    source/geometry/volumes/include/G4NavigationHistory.icc

     98 inline
     99 const G4AffineTransform& G4NavigationHistory::GetTopTransform() const
    100 {
    101   return fNavHistory[fStackDepth].GetTransform();
    102 }


     source/geometry/volumes/include/G4NavigationLevel.hh

     54 class G4NavigationLevel
     55 {
     56 
     57  public:  // with description
     58 
     59    G4NavigationLevel(G4VPhysicalVolume*       newPtrPhysVol,
     60                      const G4AffineTransform& newT,
     61                      EVolume                  newVolTp,
     62                      G4int                    newRepNo= -1);
     63 
     64    G4NavigationLevel(G4VPhysicalVolume*       newPtrPhysVol,
     65                      const G4AffineTransform& levelAbove,
     66                      const G4AffineTransform& relativeCurrent,
     67                      EVolume                  newVolTp,
     68                      G4int                    newRepNo= -1);
     69      // As the previous constructor, but instead of giving Transform, give 
     70      // the AffineTransform to the level above and the current level's 
     71      // Transform relative to that.
     72 
     73    G4NavigationLevel();
     74    G4NavigationLevel( const G4NavigationLevel& );
     75 
     76    ~G4NavigationLevel();
     77 
     78    G4NavigationLevel& operator=(const G4NavigationLevel &right);
     79 
     80    inline G4VPhysicalVolume*       GetPhysicalVolume() const;
     81    inline const G4AffineTransform* GetTransformPtr() const ;  // New
     82    inline const G4AffineTransform& GetTransform() const ;     // Old
     83 

     source/geometry/volumes/include/G4NavigationLevelRep.hh

     84    inline const G4AffineTransform& GetTransform() const ;     // Old
     85 
     86    inline EVolume            GetVolumeType() const ;
     87    inline G4int              GetReplicaNo() const ;
     88 
     ..  
     98  private:
     99 
    100    G4AffineTransform  sTransform;
    101      // Compounded global->local transformation (takes a point in the 
    102      // global reference system to the system of the volume at this level)





Adding Detection Surfaces
----------------------------

Using a surface per-PMT exceeds a limit 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daedirectpropagator.py", line 133, in main
        cpl_end = propagator.propagate(cpl_begin) 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daedirectpropagator.py", line 48, in propagate
        max_steps=max_steps)
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/photon.py", line 145, in propagate
        if ga.max(self.flags).get() & (1 << 31):
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/gpuarray.py", line 1249, in f
        krnl = get_minmax_kernel(what, a.dtype)
      File "<string>", line 2, in get_minmax_kernel
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/tools.py", line 404, in context_dependent_memoize
        result = func(*args)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/reduction.py", line 393, in get_minmax_kernel
        }, preamble="#define MY_INFINITY (1./0)")
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/reduction.py", line 206, in __init__
        preamble=preamble)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/reduction.py", line 184, in get_reduction_kernel_and_types
        func = mod.get_function(name)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/compiler.py", line 285, in get_function
        return self.module.get_function(name)
    pycuda._driver.LaunchError: cuModuleGetFunction failed: launch failed
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuModuleUnload failed: launch failed
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuMemFree failed: launch failed
    PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    cuMemFree failed: launch failed


::

    delta:chroma blyth$ collada_to_chroma.sh

    In [2]: len(cg.unique_surfaces)
    Out[2]: 714



An obvious limitation is from chroma/gpu/geometry.py::

    144 
    145         material_codes = (((geometry.material1_index & 0xff) << 24) |
    146                           ((geometry.material2_index & 0xff) << 16) |
    147                           ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)


As unique identifiers are squeezed into one byte::

    In [1]: 0xff
    Out[1]: 255


::

     54 struct Geometry
     55 {
     56     float3 *vertices;
     57     uint3 *triangles;
     58     unsigned int *material_codes;
     59     unsigned int *colors;
     60     uint4 *primary_nodes;
     61     uint4 *extra_nodes;
     62     Material **materials;
     63     Surface **surfaces;
     64     float3 world_origin;
     65     float world_scale;
     66     int nprimary_nodes;
     67 };





Questions 
-----------

#. How hits are handled in Chroma ?



chroma/geometry.py
---------------------

::

        103 class Solid(object):
        104     """Solid object attaches materials, surfaces, and colors to each triangle
        105     in a Mesh object."""
        106     def __init__(self, mesh, material1=None, material2=None, surface=None, color=0x33ffffff):
        107         self.mesh = mesh


        /// collection of solids, maybe with a material flagged as a detector_material
        /// Nope: detector_material appears unused

        261 class Geometry(object):
        262     "Geometry object."
        263     def __init__(self, detector_material=None):
        264         self.detector_material = detector_material
        265         self.solids = []
        266         self.solid_rotations = []
        267         self.solid_displacements = []
        268         self.bvh = None

        ///
        /// to get detection, need to model as a surface with detect property ?
        ///

        234 class Surface(object):
        235     """Surface optical properties."""
        236     def __init__(self, name='none', model=0):
        237         self.name = name
        238         self.model = model
        239 
        240         self.set('detect', 0)
        241         self.set('absorb', 0)
        242         self.set('reemit', 0)
        243         self.set('reflect_diffuse', 0)
        244         self.set('reflect_specular', 0)
        245         self.set('eta', 0)
        246         self.set('k', 0)
        247         self.set('reemission_cdf', 0)
        248 
        249         self.thickness = 0.0
        250         self.transmissive = 0
        251 
        252     def set(self, name, value, wavelengths=standard_wavelengths):
        253         if np.iterable(value):
        254             if len(value) != len(wavelengths):
        255                 raise ValueError('shape mismatch')
        256         else:
        257             value = np.tile(value, len(wavelengths))
        258 
        259         if (np.asarray(value) < 0.0).any():
        260             raise Exception('all probabilities must be >= 0.0')
        261 
        262         self.__dict__[name] = np.array(zip(wavelengths, value), dtype=np.float32)
        263     def __repr__(self):
        264         return '<Surface %s>' % self.name


detector_material not used ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (chroma_env)delta:chroma blyth$ find . -type f -exec grep -H detector_material {} \;
    ./chroma/detector.py:    def __init__(self, detector_material=None):
    ./chroma/detector.py:        Geometry.__init__(self, detector_material=detector_material)
    ./chroma/geometry.py:    def __init__(self, detector_material=None):
    ./chroma/geometry.py:        self.detector_material = detector_material
    ./chroma/sim.py:            self.photon_generator = generator.photon.G4ParallelGenerator(geant4_processes, detector.detector_material, base_seed=self.seed)


#. Chroma looks to expect sensdet to be represented as surfaces 
   as needed to get a SURFACE_DETECT ? 


detsim does EFFICIENCY lookup on Bialkali materialof the cathode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/hemi-pmt.xml`::

    118   <!-- The Photo Cathode -->
    119   <!-- use if limit photocathode to a face on diameter gt 167mm. -->
    120   <logvol name="lvPmtHemiCathode" material="Bialkali" sensdet="DsPmtSensDet">
    121     <union name="pmt-hemi-cathode">
    122       <sphere name="pmt-hemi-cathode-face"
    123           outerRadius="PmtHemiFaceROCvac"
    124           innerRadius="PmtHemiFaceROCvac-PmtHemiCathodeThickness"
    125           deltaThetaAngle="PmtHemiFaceCathodeAngle"/>
    126       <sphere name="pmt-hemi-cathode-belly"
    127           outerRadius="PmtHemiBellyROCvac"
    128           innerRadius="PmtHemiBellyROCvac-PmtHemiCathodeThickness"
    129           startThetaAngle="PmtHemiBellyCathodeAngleStart"
    130           deltaThetaAngle="PmtHemiBellyCathodeAngleDelta"/>
    131       <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
    132     </union>
    133   </logvol>
    134   <!-- use if limit photocathode to a face on diameter lt 167mm. -->
    135   <!-- 
    136   <logvol name="lvPmtHemiCathode" material="Bialkali" sensdet="DsPmtSensDet">
    137     <sphere name="pmt-hemi-cathode-face"
    138         outerRadius="PmtHemiFaceROCvac"
    139         innerRadius="PmtHemiFaceROCvac-PmtHemiCathodeThickness"
    140         deltaThetaAngle="PmtHemiFaceCathodeAngle"/>
    141   </logvol>


`NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/headon-pmt.xml`::

     72   <!-- The Photo Cathode -->
     73   <logvol name="lvHeadonPmtCathode" material="Bialkali" sensdet="DsPmtSensDet">
     74     <tubs name="headon-pmt-cath"
     75           sizeZ="HeadonPmtCathodeThickness"
     76       outerRadius="HeadonPmtGlassRadius-HeadonPmtGlassWallThick"/>
     77   </logvol>


`NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/materials/bialkali.xml`::

     07   <catalog name="BialkaliProperties">
      8 
      9 
     10     <!-- From G4dyb -->
     11     <tabproperty name="PhotoCathodeQE"
     12          type="EFFICIENCY"
     13          xunit="eV"
     14          yunit=""
     15          xaxis="PhotonEnergy"
     16          yaxis="QuantumEfficiency">
     17                  1.55000                 0.00010
     18                  1.80000                 0.00200
     19                  1.90000                 0.00500
     ..
     38                  2.99000                 0.22000
     39                  3.06000                 0.22000
     40                  3.14000                 0.23000
     41                  3.22000                 0.24000
     42                  3.31000                 0.24000
     43                  3.40000                 0.24000
     44                  3.49000                 0.23000
     45                  3.59000                 0.22000
     46                  3.70000                 0.21000
     47                  3.81000                 0.17000
     48                  3.94000                 0.14000
     49                  4.07000                 0.09000
     50                  4.10000                 0.03500
     ..
     56     </tabproperty>
     ..
     59     <!-- From G4dyb -->
     60     <tabproperty name="PhotoCathodeRefractionIndex"
     61          type="RINDEX"
     62          xunit="eV"
     63          xaxis="PhotonEnergy"
     64          yaxis="RefractionIndex">
     65     1.55    2.9
     66     6.20    2.9
     67     10.33   2.9
     68     15.5    2.9
     69     </tabproperty>
     70 
     71     <tabproperty name="PhotoCathodeImaginaryIndex"
     72          type="KINDEX"
     73          xunit="eV"
     74          xaxis="PhotonEnergy"
     75          yaxis="RefractionIndex">
     76     1.55    1.6
     77     6.20    1.6
     78     10.33   1.6
     79     15.5    1.6
     80     </tabproperty>


::

    [blyth@belle7 dybgaudi]$ find . -name '*.cc' -exec grep -H RINDEX {} \;
    ./Simulation/DetSim/src/DsG4OpRayleigh.cc:        G4MaterialPropertyVector* Rindex = aMPT->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:       Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:           theStatus = NoRINDEX;
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:           theStatus = NoRINDEX;
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                  Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:            theStatus = NoRINDEX;
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                      aMaterialPropertiesTable->GetProperty("REALRINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                      aMaterialPropertiesTable->GetProperty("IMAGINARYRINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                 Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:        theStatus = NoRINDEX;
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                if ( theStatus == NoRINDEX )
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                        G4cout << " *** NoRINDEX *** " << G4endl;
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:                aMaterialPropertiesTable->GetProperty("RINDEX"); 
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:                   aMaterialPropertiesTable->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:                     Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
    ./DybAlg/src/components/DybModifyProperties.cc:   // v.push_back("RINDEX");
    ./Reconstruction/PoolMuonRec/src/PoolMuonRecTool.cc:    if(type=="RINDEX") {
    [blyth@belle7 dybgaudi]$ 
    [blyth@belle7 dybgaudi]$ find . -name '*.cc' -exec grep -H KINDEX {} \;
    [blyth@belle7 dybgaudi]$ 




`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.cc`::

    056 DsPmtSensDet::DsPmtSensDet(const std::string& type,
     57                            const std::string& name, 
     58                            const IInterface*  parent)
     59     : G4VSensitiveDetector(name)
     60     , GiGaSensDetBase(type,name,parent)
     61     , m_t2de(0)
     62 {
     63     info() << "DsPmtSensDet (" << type << "/" << name << ") created" << endreq;
     64 
     65     declareProperty("CathodeLogicalVolume",
     66                     m_cathodeLogVols,
     67                     "Photo-Cathode logical volume to which this SD is attached.");
     68 
     69     declareProperty("TouchableToDetelem", m_t2deName = "TH2DE",
     70                     "The ITouchableToDetectorElement to use to resolve sensor.");
     71 
     72     declareProperty("SensorStructures",m_sensorStructures,
     73                     "TDS Paths in which to look for sensor detector elements"
     74                     " using this sensitive detector");
     75 
     76     declareProperty("PackedIdPropertyName",m_idParameter="PmtID",
     77                     "The name of the user property holding the PMT ID.");
     78 
     79     declareProperty("QEffParameterName",m_qeffParamName="EFFICIENCY",
     80                     "name of user parameter in the photo cathode volume that"
     81                     " holds the quantum efficiency tabproperty");
     82 
     83     declareProperty("QEScale",m_qeScale=1.0 / 0.9,
     84                     "Upward scaling of the quantum efficiency by inverse of mean PMT-to-PMT efficiency in electronics simulation.");
     85 
     86     declareProperty("ConvertWeightToEff", m_ConvertWeightToEff=false,
     87                     "Treat to the optical photon weight as to preliminary applied QE."
     88                     "Will affect only the primary photons (GtDiffuserBallTool, etc.).");
     89 
     90     m_cathodeLogVols.push_back("/dd/Geometry/PMT/lvPmtHemiCathode");
     91     m_cathodeLogVols.push_back("/dd/Geometry/PMT/lvHeadonPmtCathode");
     92 }
     93 



    277 double DsPmtSensDet::SensDetQE(G4LogicalVolume* logvol, double energy)
    278 {
    279     G4Material* mat = logvol->GetMaterial();
    280     if (!mat) {
    281         warning () << "No material for " << logvol->GetName() << endreq;
    282         return -1;
    283     }
    284 
    285 
    286      G4MaterialPropertiesTable* mattab = mat->GetMaterialPropertiesTable();
    287     if (mattab) {
    288         G4MaterialPropertyVector* qevec = mattab->GetProperty(m_qeffParamName.c_str());
    289         if (qevec) {
    290 
    291       verbose() << m_qeffParamName << ":("
    292           << qevec->GetMinPhotonEnergy()/CLHEP::eV << " eV,"
    293           << qevec->GetMinProperty() << ")-->("
    294           << qevec->GetMaxPhotonEnergy()/CLHEP::eV << " eV,"
    295           << qevec->GetMaxProperty() << ")"
    296           << " particle energy is " << energy/CLHEP::eV
    297           << endreq;
    298 
    299       return qevec->GetProperty(energy);
    300 
    301         }
    302     }
    303     else {
    304         debug () << "No material properties in " << logvol->GetName() << endreq;
    305     }
    306 
    307     int ndaught = logvol->GetNoDaughters();
    308     for (int ind=0; ind < ndaught; ++ind) {
    309         G4VPhysicalVolume* physvol = logvol->GetDaughter(ind);
    310         double qe = this->SensDetQE(physvol->GetLogicalVolume(),energy);
    311         if (qe < 0) return qe;
    312     }
    313     warning() << "All attempts failed to find " << m_qeffParamName
    314               << " in " << logvol->GetName() << endreq;
    315     return -1;
    316 }



cathode example
~~~~~~~~~~~~~~~~

`chroma/demo/optics.py`::

     26 # r7081hqe photocathode material surface
     27 # source: hamamatsu supplied datasheet for r7081hqe pmt serial number zd0062
     28 r7081hqe_photocathode = Surface('r7081hqe_photocathode')
     29 r7081hqe_photocathode.detect = \
     30     np.array([(260.0,  0.00),
     31               (270.0,  0.04), (280.0,  0.07), (290.0,  0.77), (300.0,  4.57),
     32               (310.0, 11.80), (320.0, 17.70), (330.0, 23.50), (340.0, 27.54),
     33               (350.0, 30.52), (360.0, 31.60), (370.0, 31.90), (380.0, 32.20),
     34               (390.0, 32.00), (400.0, 31.80), (410.0, 30.80), (420.0, 30.16),
     35               (430.0, 29.24), (440.0, 28.31), (450.0, 27.41), (460.0, 26.25),
     36               (470.0, 24.90), (480.0, 23.05), (490.0, 21.58), (500.0, 19.94),
     37               (510.0, 18.48), (520.0, 17.01), (530.0, 15.34), (540.0, 12.93),
     38               (550.0, 10.17), (560.0,  7.86), (570.0,  6.23), (580.0,  5.07),
     39               (590.0,  4.03), (600.0,  3.18), (610.0,  2.38), (620.0,  1.72),
     40               (630.0,  0.95), (640.0,  0.71), (650.0,  0.44), (660.0,  0.25),
     41               (670.0,  0.14), (680.0,  0.07), (690.0,  0.03), (700.0,  0.02),
     42               (710.0,  0.00)])
     43 # convert percent -> fraction
     44 r7081hqe_photocathode.detect[:,1] /= 100.0
     45 # roughly the same amount of detected photons are absorbed without detection
     46 r7081hqe_photocathode.absorb = r7081hqe_photocathode.detect
     47 # remaining photons are diffusely reflected
     48 r7081hqe_photocathode.set('reflect_diffuse', 1.0 - r7081hqe_photocathode.detect[:,1] - r7081hqe_photocathode.absorb[:,1], wavelengths=r7081hqe_photocathode.detect[:,0])
     49 


chroma/detector.py
-------------------

::

     05 class Detector(Geometry):
     06     '''A Detector is a subclass of Geometry that allows some Solids
     07     to be marked as photon detectors, which we will suggestively call
     08     "PMTs."  Each detector is imagined to be connected to an electronics
     09     channel that records a hit time and charge.
     10 
     11     Each PMT has two integers identifying it: a channel index and a
     12     channel ID.  When all of the channels in the detector are stored
     13     in a Numpy array, they will be stored in index order.  Channel
     14     indices star from zero and have no gaps.  Channel ID numbers are
     15     arbitrary integers that identify a PMT uniquely in a stable way.
     16     They are written out to disk when using the Chroma ROOT format,
     17     and are used when reading events back in to map channels back
     18     into the correct array index.
     19 
     20     For now, all the PMTs share a single set of time and charge
     21     distributions.  In the future, this will be generalized to
     22     allow per-channel distributions.
     23     '''
     24 
     25     def __init__(self, detector_material=None):
     26         Geometry.__init__(self, detector_material=detector_material)
     27 
     28         # Using numpy arrays here to allow for fancy indexing
     29         self.solid_id_to_channel_index = np.zeros(0, dtype=np.int32)
     30         self.channel_index_to_solid_id = np.zeros(0, dtype=np.int32)
     31 
     32         self.channel_index_to_channel_id = np.zeros(0, dtype=np.int32)
     33 
     34         # If the ID numbers are arbitrary, we can't treat them
     35         # as array indices, so have to use a dictionary
     36         self.channel_id_to_channel_index = {}
     37 
     38         # zero time and unit charge distributions
     39         self.time_cdf = (np.array([-0.01, 0.01]), np.array([0.0, 1.0]))
     40         self.charge_cdf = (np.array([0.99, 1.00]), np.array([0.0, 1.0]))


add_solid/add_pmt
~~~~~~~~~~~~~~~~~~~~

When have a `channel_id` associated to a volume  **add_pmt**  is used instead of **add_solid**.
The idmap adds `channel_id` attribute to some DAENode. Which 
in geometry conversion to chroma results in use of *add_pmt* rather than *add_solid* 

chroma.detector.add_pmt::

     54     def add_pmt(self, pmt, rotation=None, displacement=None, channel_id=None):
     55         """Add the PMT `pmt` to the geometry. When building the final triangle
     56         mesh, `solid` will be placed by rotating it with the rotation matrix
     57         `rotation` and displacing it by the vector `displacement`, just like
     58         add_solid().
     59 
     60             `pmt``: instance of chroma.Solid
     61                 Solid representing a PMT.
     62             `rotation`: numpy.matrix (3x3)
     63                 Rotation to apply to PMT mesh before displacement.  Defaults to
     64                 identity rotation.
     65             `displacement`: numpy.ndarray (shape=3)
     66                 3-vector displacement to apply to PMT mesh after rotation.
     67                 Defaults to zero vector.
     68             `channel_id`: int
     69                 Integer identifier for this PMT.  May be any integer, with no
     70                 requirement for consective numbering.  Defaults to None,
     71                 where the ID number will be set to the generated channel index.
     72                 The channel_id must be representable as a 32-bit integer.
     73         
     74             Returns: dictionary { 'solid_id' : solid_id, 
     75                                   'channel_index' : channel_index,
     76                                   'channel_id' : channel_id }
     77         """
     .. 
     82         channel_index = len(self.channel_index_to_solid_id)
     83         if channel_id is None:
     84             channel_id = channel_index
     85 
     86         # add_solid resized this array already
     87         self.solid_id_to_channel_index[solid_id] = channel_index
     88 
     89         # resize channel_index arrays before filling
     90         self.channel_index_to_solid_id.resize(channel_index+1)
     91         self.channel_index_to_solid_id[channel_index] = solid_id
     92         self.channel_index_to_channel_id.resize(channel_index+1)
     93         self.channel_index_to_channel_id[channel_index] = channel_id
     94 
     95         # dictionary does not need resizing
     96         self.channel_id_to_channel_index[channel_id] = channel_index
       
chroma.detector.add_solid::

     50         self.solid_id_to_channel_index.resize(solid_id+1)
     51         self.solid_id_to_channel_index[solid_id] = -1 # solid maps to no channel


chroma/gpu/detector.py
-----------------------

To get the mapping copied to GPU need to use GPUDetector rather than GPUGeometry::

     14 class GPUDetector(GPUGeometry):
     15     def __init__(self, detector, wavelengths=None, print_usage=False):
     16         GPUGeometry.__init__(self, detector, wavelengths=wavelengths, print_usage=False)
     17         self.solid_id_to_channel_index_gpu = \
     18             ga.to_gpu(detector.solid_id_to_channel_index.astype(np.int32))
     19         self.nchannels = detector.num_channels()
     20 
     21 
     22         self.time_cdf_x_gpu = ga.to_gpu(detector.time_cdf[0].astype(np.float32))
     23         self.time_cdf_y_gpu = ga.to_gpu(detector.time_cdf[1].astype(np.float32))
     24 
     25         self.charge_cdf_x_gpu = ga.to_gpu(detector.charge_cdf[0].astype(np.float32))
     26         self.charge_cdf_y_gpu = ga.to_gpu(detector.charge_cdf[1].astype(np.float32))
     27 
     28         detector_source = get_cu_source('detector.h')
     29         detector_struct_size = characterize.sizeof('Detector', detector_source)
     30         self.detector_gpu = make_gpu_struct(detector_struct_size,
     31                                             [self.solid_id_to_channel_index_gpu,
     32                                              self.time_cdf_x_gpu,
     33                                              self.time_cdf_y_gpu,
     34                                              self.charge_cdf_x_gpu,
     35                                              self.charge_cdf_y_gpu,
     36                                              np.int32(self.nchannels),
     37                                              np.int32(len(detector.time_cdf[0])),
     38                                              np.int32(len(detector.charge_cdf[0])),
     39                                              np.float32(detector.charge_cdf[0][-1] / 2**16)])


Crucial connection between solids and channels, handled in **solid_id_to_channel_index[solid_id]**.
This distinquishes sensitive solids (PMTs).::

    simon:chroma blyth$ find . -name '*.*' -exec grep -H solid_id_to_channel_index {} \;
    ./cuda/daq.cu:      int channel_index = detector->solid_id_to_channel_index[solid_id];
    ./cuda/daq.cu:      channel_index = detector->solid_id_to_channel_index[solid_id];
    ./cuda/detector.h:    int *solid_id_to_channel_index;
    ./detector.py:        self.solid_id_to_channel_index = np.zeros(0, dtype=np.int32)
    ./detector.py:        self.solid_id_to_channel_index.resize(solid_id+1)
    ./detector.py:        self.solid_id_to_channel_index[solid_id] = -1 # solid maps to no channel
    ./detector.py:        self.solid_id_to_channel_index[solid_id] = channel_index
    ./gpu/daq.py:        self.solid_id_to_channel_index_gpu = gpu_detector.solid_id_to_channel_index_gpu
    ./gpu/detector.py:        self.solid_id_to_channel_index_gpu = \
    ./gpu/detector.py:            ga.to_gpu(detector.solid_id_to_channel_index.astype(np.int32))
    ./gpu/detector.py:                                            [self.solid_id_to_channel_index_gpu,




chroma/gpu/daq.py
-------------------

::

     60     def acquire(self, gpuphotons, rng_states, nthreads_per_block=64, max_blocks=1024, start_photon=None, nphotons=None, weight=1.0):
     61         if start_photon is None:
     62             start_photon = 0
     63         if nphotons is None:
     64             nphotons = len(gpuphotons.pos) - start_photon
     65 
     66         if self.ndaq == 1:
     67             for first_photon, photons_this_round, blocks in \
     68                     chunk_iterator(nphotons, nthreads_per_block, max_blocks):
     69                 self.gpu_funcs.run_daq(rng_states, np.uint32(0x1 << 2),
     70                                        np.int32(start_photon+first_photon), np.int32(photons_this_round), gpuphotons.t,
     71                                        gpuphotons.flags, gpuphotons.last_hit_triangles, gpuphotons.weights,
     72                                        self.solid_id_map_gpu,
     73                                        self.detector_gpu,
     74                                        self.earliest_time_int_gpu,
     75                                        self.channel_q_int_gpu, self.channel_history_gpu,
     76                                        np.float32(weight),
     77                                        block=(nthreads_per_block,1,1), grid=(blocks,1))



run_daq
~~~~~~~~~

Uses atomics to do histogramming, and find earliest time.


`chroma/chroma/cuda/daq.cu`::

     35 __global__ void
     36 run_daq(curandState *s, unsigned int detection_state,
     37     int first_photon, int nphotons, float *photon_times,
     38     unsigned int *photon_histories, int *last_hit_triangles,
     39     float *weights,
     40     int *solid_map,
     41     Detector *detector,
     42     unsigned int *earliest_time_int,
     43     unsigned int *channel_q_int, unsigned int *channel_histories,
     44     float global_weight)
     45 {
     46 
     47     int id = threadIdx.x + blockDim.x * blockIdx.x;
     48 
     49     if (id < nphotons) {
     50     curandState rng = s[id];
     51     int photon_id = id + first_photon;
     52     int triangle_id = last_hit_triangles[photon_id];
     53 
     54     if (triangle_id > -1) {
     55         int solid_id = solid_map[triangle_id];
     56         unsigned int history = photon_histories[photon_id];
     57         int channel_index = detector->solid_id_to_channel_index[solid_id];
     58
     59         if (channel_index >= 0 && (history & detection_state)) {
     60 
     61         float weight = weights[photon_id] * global_weight;
     62         if (curand_uniform(&rng) < weight) {
     63             float time = photon_times[photon_id] +
     64                       sample_cdf(&rng, detector->time_cdf_len,
     65                                        detector->time_cdf_x, detector->time_cdf_y);
     //
     66             unsigned int time_int = float_to_sortable_int(time);
     67 
     68             float charge = sample_cdf(&rng, detector->charge_cdf_len,
     69                                             detector->charge_cdf_x,
     70                                             detector->charge_cdf_y);
     //
     71             unsigned int charge_int = roundf(charge / detector->charge_unit);
     72 
     73             atomicMin(earliest_time_int + channel_index, time_int);
     74             atomicAdd(channel_q_int + channel_index, charge_int);
     75             atomicOr(channel_histories + channel_index, history);
     76         } // if weighted photon contributes
     77 
     78         } // if photon detected by a channel
     79 
     80     } // if photon terminated on surface
     81 
     82     s[id] = rng;
     83 
     84     }


Hmm is the CDF sampling here equivalent to QE handling in ProcessHits ?

* no, seems not : the equivalent is done in ElecSim presumably 


SURFACE_DETECT
----------------

::

     18 struct Surface
     19 {
     20     float *detect;
     21     float *absorb;
     22     float *reemit;
     23     float *reflect_diffuse;
     24     float *reflect_specular;
     25     float *eta;
     26     float *k;
     27     float *reemission_cdf;
     28 
     29     unsigned int model;
     30     unsigned int n;
     31     unsigned int transmissive;
     32     float step;
     33     float wavelength0;
     34     float thickness;
     35 };


Looks like equivent of ProcessHits QE is to set, the surface detect property.


chroma/cuda/photon.h 
---------------------

::

     47 enum
     48 {
     49     NO_HIT           = 0x1 << 0,
     50     BULK_ABSORB      = 0x1 << 1,
     51     SURFACE_DETECT   = 0x1 << 2,
     52     SURFACE_ABSORB   = 0x1 << 3,
     53     RAYLEIGH_SCATTER = 0x1 << 4,
     54     REFLECT_DIFFUSE  = 0x1 << 5,
     55     REFLECT_SPECULAR = 0x1 << 6,
     56     SURFACE_REEMIT   = 0x1 << 7,
     57     SURFACE_TRANSMIT = 0x1 << 8,
     58     BULK_REEMIT      = 0x1 << 9,
     59     NAN_ABORT        = 0x1 << 31
     60 }; // processes

::

    In [16]: np.uint32(0x1 << 2)
    Out[16]: 4




chroma/cuda/daq.cu
--------------------

* how do the *sample_cdf* compare with those from Geant4/DetSim ?

  * THINK there is no comparison, ElecSim code being complex moral equivalent  

Sequence::

   photon_id > triangle_id > solid_id > channel_index 


::

     35 __global__ void
     36 run_daq(curandState *s, unsigned int detection_state,
     37     int first_photon, int nphotons, float *photon_times,
     38     unsigned int *photon_histories, int *last_hit_triangles,
     39     float *weights,
     40     int *solid_map,
     41     Detector *detector,
     42     unsigned int *earliest_time_int,
     43     unsigned int *channel_q_int, unsigned int *channel_histories,
     44     float global_weight)
     45 {
     46 
     47     int id = threadIdx.x + blockDim.x * blockIdx.x;
     48 
     49     if (id < nphotons) {
     50     curandState rng = s[id];
     51     int photon_id = id + first_photon;
     52     int triangle_id = last_hit_triangles[photon_id];
     53 
     54     if (triangle_id > -1) {
     55         int solid_id = solid_map[triangle_id];
     56         unsigned int history = photon_histories[photon_id];
     57         int channel_index = detector->solid_id_to_channel_index[solid_id];
     58 
     59         if (channel_index >= 0 && (history & detection_state)) {                  // SURFACE_DETECT flagged in history   
     60 
     61         float weight = weights[photon_id] * global_weight;
     62         if (curand_uniform(&rng) < weight) {
     63             float time = photon_times[photon_id] +
     64             sample_cdf(&rng, detector->time_cdf_len,
     65                    detector->time_cdf_x, detector->time_cdf_y);
     66             unsigned int time_int = float_to_sortable_int(time);
     67 
     68             float charge = sample_cdf(&rng, detector->charge_cdf_len,
     69                       detector->charge_cdf_x,
     70                       detector->charge_cdf_y);
     71             unsigned int charge_int = roundf(charge / detector->charge_unit);
     72 
     73             atomicMin(earliest_time_int + channel_index, time_int);
     74             atomicAdd(channel_q_int + channel_index, charge_int);
     75             atomicOr(channel_histories + channel_index, history);
     76         } // if weighted photon contributes
     77 
     78         } // if photon detected by a channel
     79 
     80     } // if photon terminated on surface
     81 
     82     s[id] = rng;
     83    
     84     }
     85    
     86 }



