Chroma Modeling of Sensitive Detectors 
=========================================

.. contents:: :local:

Overview
--------

#. Chroma/DetSim mismatch: 

   * DetSim, QE handled as a Bialkali material EFFICIENCY property
   * Chroma, **Surface.detect** property 

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

When have a `channel_id` associated to a volume can use **add_pmt**  instead of **add_solid**

::

     42     def add_solid(self, solid, rotation=None, displacement=None):
     43         """
     44         Add the solid `solid` to the geometry. When building the final triangle
     45         mesh, `solid` will be placed by rotating it with the rotation matrix
     46         `rotation` and displacing it by the vector `displacement`.
     47         """
     48         solid_id = Geometry.add_solid(self, solid=solid, rotation=rotation,
     49                                       displacement=displacement)
     50         self.solid_id_to_channel_index.resize(solid_id+1)
     51         self.solid_id_to_channel_index[solid_id] = -1 # solid maps to no channel
     52         return solid_id
     53 
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
     78 
     79         solid_id = self.add_solid(solid=pmt, rotation=rotation,
     80                                   displacement=displacement)
     81 
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
     97 
     98         return { 'solid_id' : solid_id,
     99                  'channel_index' : channel_index,
     00                  'channel_id' : channel_id }




chroma/gpu/detector.py
-----------------------

::

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

* how do the *sample_cdf* compare with those from Geant4 ?

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



