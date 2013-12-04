Memory Corruption
=================

Running with 

``MALLOC_CHECK_=1``
        keep going even in encountering corruption, in order to establish context
``LIBC_FATAL_STDERR_=1``
        write malloc debug output to stderr rather than devtty   


Implicates VRML scenehandler creation.::

    779 GiGaRunActionGDML::BeginOfRunAction i 0 c V
    780 GiGaRunActionGDML::WriteVis vis open
    781 /vis/sceneHandler/create VRML2FILE
    782 G4VisManager::SetCurrentGraphicsSystem: system now VRML2FILE
    783 Graphics system set to VRML2FILE
    784 New scene handler "scene-handler-0" created.
    785 *** glibc detected *** python: malloc(): memory corruption: 0x0f5f63f8 ***
    786 /vis/viewer/create ! ! 600
    787 *** glibc detected *** python: malloc: top chunk is corrupt: 0x0f5f63f0 ***
    788 G4VisManager::CreateViewer: new viewer created.
    789  view parameters are:
    790   View parameters and options:
    791   Drawing style: wireframe

    1886 G4GDML: Writing '/data1/env/local/env/geant4/geometry/gdml/VDGVDGX_20131204-1252/g4_00.gdml' done !
    1887 GiGaRunActionGDML::BeginOfRunAction i 3 c V
    1888 GiGaRunActionGDML::WriteVis vis open
    1889 /vis/sceneHandler/create VRML2FILE
    1890 G4VisManager::SetCurrentGraphicsSystem: system now VRML2FILE
    1891 Graphics system set to VRML2FILE
    1892 New scene handler "scene-handler-1" created.
    1893 /vis/sceneHandler/attach
    1894 *** glibc detected *** python: malloc(): memory corruption: 0x0f608c80 ***
    1895 Scene "scene-0" attached to scene handler "scene-handler-1.
    1896   (You may have to refresh with "/vis/viewer/flush" if view is not "auto-refresh".)
    1897 *** glibc detected *** python: double free or corruption (out): 0x0fe3c1a8 ***
    1898 *** glibc detected *** python: malloc: top chunk is corrupt: 0x0f608c78 ***
    1899 /vis/viewer/create ! ! 600
    1900 G4VisManager::CreateViewer: new viewer created.
    1901  view parameters are:
    1902   View parameters and options:
    1903   Drawing style: wireframe
    1904   Auxiliary edges: invisible


#. Exporting DAE only ``export.sh D`` shows no corruption.

#. With VRML alone ``export.sh V`` get the corruption::

     779 GiGaRunActionGDML::BeginOfRunAction i 0 c V
     780 GiGaRunActionGDML::WriteVis vis open
     781 /vis/sceneHandler/create VRML2FILE
     782 G4VisManager::SetCurrentGraphicsSystem: system now VRML2FILE
     783 Graphics system set to VRML2FILE
     784 *** glibc detected *** python: malloc(): memory corruption: 0x0e3f22b8 ***
     785 New scene handler "scene-handler-0" created.
     786 /vis/viewer/create ! ! 600
     787 G4VisManager::CreateViewer: new viewer created.
     788  view parameters are:




