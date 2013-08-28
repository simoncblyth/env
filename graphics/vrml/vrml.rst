VRML
=====

.. contents:: :local:

Versions
-----------

* http://en.wikipedia.org/wiki/VRML

In 1997, a new version of the format was finalized, as VRML97 (also known as VRML2 or VRML 2.0), and became an ISO standard.


Export From Geant4
---------------------

VRML is not too distant from Geant4 geometry representation as collection of C++ instances of solid representing classes. 
Thus Geant4 to VRML conversion should not be too difficult. At least the code looks straightforward::

    [blyth@cms01 src]$ vi G4VRML2SceneHandlerFunc.icc
    [blyth@cms01 src]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src


* http://geant4.kek.jp/GEANT4/vis/GEANT4/VRML_file_driver.html
* http://geant4.slac.stanford.edu/Presentations/vis/G4VisCommands.pdf
* http://geant4.slac.stanford.edu/Presentations/vis/G4DAWNTutorial/G4DAWNTutorial.html

Opening a driver eg OGL,DAWNFILE,HepRepFile,VRML2FILE.
Extrapolating from DAWNFILE tutorial the macro commands to export detector geometry
(not hits etc) are probably::

    # Use this open statement to create a .wrl file suitable for viewing in a VRML viewer
    /vis/open VRML2FILE 
    /vis/drawVolume
    /vis/viewer/flush  


VRML Viewers
-------------

* http://www.web3d.org/x3d/vrml/tools/viewers_and_browsers/

InstantReality
~~~~~~~~~~~~~~~

* http://doc.instantreality.org/documentation/getting-started/
* http://doc.instantreality.org/media/uploads/downloads/2.3.0/InstantPlayer-MacOS-10.4-universal-2.3.0.25322.dmg
* http://www.instantreality.org/story/modules/
* http://doc.instantreality.org/tutorial/installation-on-mac-os-x/


::

    simon:geant4 blyth$ cd ~/Desktop
    simon:Desktop blyth$ curl -L -O http://doc.instantreality.org/media/uploads/downloads/2.3.0/InstantPlayer-MacOS-10.4-universal-2.3.0.25322.dmg
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  148M  100  148M    0     0   167k      0  0:15:06  0:15:06 --:--:--  183k



Viewing the .wrl geometry is painfully slow, and brings molasses to entire machine UI. But something is visible.


FreeWRL
~~~~~~~~~

* http://freewrl.sourceforge.net/

Build issues.


Blender
~~~~~~~~

* http://www.blender.org/forum/viewtopic.php?t=12402
* http://sourceforge.net/projects/vrml97import/







