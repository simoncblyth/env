Introduction Resources
=======================


* http://geant4.cern.ch/geant4/collaboration/working_groups/geometry/training/D2-Basics.pdf


Solid LV PV
-------------

* LV is everything but the position


Geometrical Hierarchy
----------------------

* Position and rotation of the daughter volume is described with 
  respect to the local coordinate system of the mother volume 

* The origin of the mother's local coordinate system is at the center of 
  the mother volume 

* Daughter volumes cannot protrude from the mother volume 

* Daughter volumes cannot overlap 


* Logical volume of mother knows the daughter volumes it contains 

  * It is uniquely defined to be their mother volume 
  * If the logical volume of the mother is placed more than once, all 
    daughters appear by definition in all these physical instances of the 
    mother 

* The origin of the global coordinate system is at the center of the 
  world volume 


Touchables
----------

A touchable for a volume serves the purpose 
of providing a unique identification for a 
detector element. Via copy numbers eg `A.3/B.2/C.1`





