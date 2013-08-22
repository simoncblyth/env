GiGa
====

::

    In [8]: from GiGa.GiGaConf import GiGa

    In [9]: giga = GiGa()

    In [10]: giga.
    Display all 110 possibilities? (y or n)
    giga.AuditFinalize                 giga.StackingAction                giga.__getattribute__              giga.__setstate__                  giga.allConfigurables              giga.getHandle                     giga.jobOptName
    giga.AuditInitialize               giga.SteppingAction                giga.__getnewargs__                giga.__sizeof__                    giga.children                      giga.getJobOptName                 giga.name
    giga.AuditReInitialize             giga.TrackingAction                giga.__getstate__                  giga.__slots__                     giga.clone                         giga.getName                       giga.printHeaderPre
    giga.AuditReStart                  giga.UIsession                     giga.__hash__                      giga.__str__                       giga.configurableServices          giga.getParent                     giga.printHeaderWidth
    giga.AuditServices                 giga.VisManager                    giga.__iadd__                      giga.__subclasshook__              giga.configurables                 giga.getPrintTitle                 giga.properties
    giga.AuditStart                    giga._Configurable__children       giga.__init__                      giga._configurationLocked          giga.copyChild                     giga.getProp                       giga.propertyNoValue
    giga.AuditStop                     giga._Configurable__setupDefaults  giga.__iter__                      giga._inSetDefaults                giga.copyChildAndSetParent         giga.getProperties                 giga.remove
    giga.DefaultName                   giga._Configurable__setupDlls      giga.__len__                       giga._initok                       giga.getAllChildren                giga.getSequence                   giga.removeAll
    giga.EventAction                   giga._Configurable__setupServices  giga.__metaclass__                 giga._isInSetDefaults              giga.getChildren                   giga.getTitleName                  giga.setDefaults
    giga.GeometrySource                giga._Configurable__tools          giga.__module__                    giga._name                         giga.getDefaultProperties          giga.getTools                      giga.setParent
    giga.OutputLevel                   giga.__class__                     giga.__new__                       giga._printFooter                  giga.getDefaultProperty            giga.getType                       giga.setProp
    giga.PhysicsList                   giga.__deepcopy__                  giga.__nonzero__                   giga._printHeader                  giga.getDlls                       giga.getValuedProperties           giga.setup
    giga.PrintG4Particles              giga.__delattr__                   giga.__reduce__                    giga._properties                   giga.getFullJobOptName             giga.hasParent                     giga.splitName
    giga.RandomNumberService           giga.__doc__                       giga.__reduce_ex__                 giga._propertyDocDct               giga.getFullName                   giga.indentUnit                    giga.toStringProperty
    giga.RunAction                     giga.__format__                    giga.__repr__                      giga._setupok                      giga.getGaudiHandle                giga.isPropertySet                 
    giga.RunManager                    giga.__getattr__                   giga.__setattr__                   giga.addTool                       giga.getGaudiType                  giga.isPublic        



ipython GiGa/Gauss introspection
----------------------------------

::


    In [1]: from GaussTools.GaussToolsConf import * 

    In [2]: filter(lambda _:'Command' in _,locals().keys())
    Out[2]: ['GiGaEventActionCommand', 'GiGaRunActionCommand', 'CommandTrackAction']

    In [3]: grac = GiGaRunActionCommand("GiGa.GiGaRunActionCommand")

    In [3]: grac.
    Display all 116 possibilities? (y or n)
    grac.AuditFinalize                 grac.PropertiesPrint               grac.__format__                    grac.__setattr__                   grac.addTool                       grac.getHandle                     grac.jobOptName
    grac.AuditInitialize               grac.RegularRowFormat              grac.__getattr__                   grac.__setstate__                  grac.allConfigurables              grac.getJobOptName                 grac.name
    grac.AuditStart                    grac.RootInTES                     grac.__getattribute__              grac.__sizeof__                    grac.children                      grac.getName                       grac.printHeaderPre
    grac.AuditStop                     grac.RootOnTES                     grac.__getnewargs__                grac.__slots__                     grac.clone                         grac.getParent                     grac.printHeaderWidth
    grac.AuditTools                    grac.StatPrint                     grac.__getstate__                  grac.__str__                       grac.configurableServices          grac.getPrintTitle                 grac.properties
    grac.BeginOfRunCommands            grac.StatTableHeader               grac.__hash__                      grac.__subclasshook__              grac.configurables                 grac.getProp                       grac.propertyNoValue
    grac.Context                       grac.TypePrint                     grac.__iadd__                      grac._configurationLocked          grac.copyChild                     grac.getProperties                 grac.remove
    grac.ContextService                grac.UseEfficiencyRowFormat        grac.__init__                      grac._inSetDefaults                grac.copyChildAndSetParent         grac.getSequence                   grac.removeAll
    grac.DefaultName                   grac._Configurable__children       grac.__iter__                      grac._initok                       grac.getAllChildren                grac.getTitleName                  grac.setDefaults
    grac.EfficiencyRowFormat           grac._Configurable__setupDefaults  grac.__len__                       grac._isInSetDefaults              grac.getChildren                   grac.getTools                      grac.setParent
    grac.EndOfRunCommands              grac._Configurable__setupDlls      grac.__metaclass__                 grac._jobOptName                   grac.getDefaultProperties          grac.getType                       grac.setProp
    grac.ErrorsPrint                   grac._Configurable__setupServices  grac.__module__                    grac._name                         grac.getDefaultProperty            grac.getValuedProperties           grac.setup
    grac.GiGaService                   grac._Configurable__tools          grac.__new__                       grac._printFooter                  grac.getDlls                       grac.hasParent                     grac.splitName
    grac.GiGaSetUpService              grac.__class__                     grac.__nonzero__                   grac._printHeader                  grac.getFullJobOptName             grac.indentUnit                    grac.toStringProperty
    grac.GlobalTimeOffset              grac.__deepcopy__                  grac.__reduce__                    grac._properties                   grac.getFullName                   grac.isInToolSvc                   
    grac.MonitorService                grac.__delattr__                   grac.__reduce_ex__                 grac._propertyDocDct               grac.getGaudiHandle                grac.isPropertySet                 
    grac.OutputLevel                   grac.__doc__                       grac.__repr__                      grac._setupok                      grac.getGaudiType                  grac.isPublic                      

 

GiGaEventActionCommand
------------------------

* source:`dybgaudi/trunk/Simulation/DetSim/python/DetSim/SetG4Verbosity.py`

::

     41         from GaussTools.GaussToolsConf import GiGaEventActionCommand
     42         geac = GiGaEventActionCommand("GiGa.GiGaEventActionCommand")
     43         geac.BeginOfEventCommands = [
     44             "/control/verbose "+str(opts.control),
     45             "/run/verbose  "+str(opts.run),
     46             "/event/verbose  "+str(opts.event),
     47             "/tracking/verbose  "+str(opts.tracking),
     48             "/geometry/navigator/verbose  "+str(opts.geometry),
     49             "/process/verbose "+str(opts.process),
     50             "/process/setVerbose " + str(opts.scint) + " Scintillation",
     51             "/process/setVerbose " + str(opts.allProcesses) + " all"
     52             ]
     53         from GiGa.GiGaConf import GiGa
     54         giga = GiGa()
     55         giga.EventAction = geac


GiGaRunActionCommand
---------------------


Try extrapolation from the EventActionCommand::

    from GaussTools.GaussToolsConf import GiGaRunActionCommand
    grac = GiGaRunActionCommand("GiGa.GiGaRunActionCommand")
    grac.BeginOfRunCommands = [
         "/vis/open VRML2FILE",
         "/vis/drawVolume",
         "/vis/viewer/flush"
    ]    
    from GiGa.GiGaConf import GiGa
    giga = GiGa()
    giga.RunAction = grac    


* http://dayabay.bnl.gov/dox/DetSim/html/visdet_8py_source.html

* NuWa-trunk/dybgaudi/Simulation/DetSim/python/visdet.py




breakpoints
------------

::

    Program received signal SIGINT, Interrupt.
    0xb6266424 in xercesc_2_8::RefHashTableOf<unsigned int>::findBucketElem () from /data1/env/local/dyb/NuWa-trunk/../external/XercesC/2.8.0/i686-slc5-gcc41-dbg/lib/libxerces-c.so.28
    (gdb) Quit
    (gdb) b 'GiGa::
    GiGa::Assert(bool, char const*, StatusCode const&) const                                                GiGa::operator>>(GiGaHitsByID&)
    GiGa::Assert(bool, std::string const&, StatusCode const&) const                                         GiGa::operator>>(GiGaHitsByName&)
    GiGa::Error(std::string const&, StatusCode const&) const                                                GiGa::prepareTheEvent(G4PrimaryVertex*)
    GiGa::Exception(std::string const&, GaudiException const&, MSG::Level const&, StatusCode const&) const  GiGa::queryInterface(InterfaceID const&, void**)
    GiGa::Exception(std::string const&, MSG::Level const&, StatusCode const&) const                         GiGa::retrieveEvent(G4Event const*&)
    GiGa::Exception(std::string const&, std::exception const&, MSG::Level const&, StatusCode const&) const  GiGa::retrieveHitCollection(GiGaHitsByID&)
    GiGa::GiGa$base(std::string const&, ISvcLocator*)                                                       GiGa::retrieveHitCollection(GiGaHitsByName&)
    GiGa::GiGa(std::string const&, ISvcLocator*)                                                            GiGa::retrieveHitCollections(G4HCofThisEvent*&)
    GiGa::Print(std::string const&, MSG::Level const&, StatusCode const&) const                             GiGa::retrieveRunManager()
    GiGa::Warning(std::string const&, StatusCode const&) const                                              GiGa::retrieveTheEvent(G4Event const*&)
    GiGa::addPrimaryKinematics(G4PrimaryVertex*)                                                            GiGa::retrieveTrajectories(G4TrajectoryContainer*&)
    GiGa::chronoSvc() const                                                                                 GiGa::rndmSvc() const
    GiGa::finalize()                                                                                        GiGa::runMgr() const
    GiGa::geoSrc() const                                                                                    GiGa::setConstruction(G4VUserDetectorConstruction*)
    GiGa::initialize()                                                                                      GiGa::setDetector(G4VPhysicalVolume*)
    GiGa::operator<<(G4PrimaryVertex*)                                                                      GiGa::setEvtAction(G4UserEventAction*)
    GiGa::operator<<(G4UserEventAction*)                                                                    GiGa::setGenerator(G4VUserPrimaryGeneratorAction*)
    GiGa::operator<<(G4UserRunAction*)                                                                      GiGa::setPhysics(G4VUserPhysicsList*)
    GiGa::operator<<(G4UserStackingAction*)                                                                 GiGa::setRunAction(G4UserRunAction*)
    GiGa::operator<<(G4UserSteppingAction*)                                                                 GiGa::setStacking(G4UserStackingAction*)
    GiGa::operator<<(G4UserTrackingAction*)                                                                 GiGa::setStepping(G4UserSteppingAction*)
    GiGa::operator<<(G4VPhysicalVolume*)                                                                    GiGa::setTracking(G4UserTrackingAction*)
    GiGa::operator<<(G4VUserDetectorConstruction*)                                                          GiGa::svcLoc() const
    GiGa::operator<<(G4VUserPhysicsList*)                                                                   GiGa::toolSvc() const
    GiGa::operator<<(G4VUserPrimaryGeneratorAction*)                                                        GiGa::~GiGa$base()
    GiGa::operator>>(G4Event const*&)                                                                       GiGa::~GiGa$delete()
    GiGa::operator>>(G4HCofThisEvent*&)                                                                     GiGa::~GiGa()
    GiGa::operator>>(G4TrajectoryContainer*&)                                                               
    (gdb) b 'GiGa::



    (gdb) b 'GiGa::initialize()'
    Breakpoint 2 at 0xb33a246f: file ../src/component/GiGa.cpp, line 145.


