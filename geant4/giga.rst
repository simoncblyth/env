GiGa
====

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


