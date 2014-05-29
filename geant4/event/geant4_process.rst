Geant4 Process
================

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch05.html

::

    What is a Process?

    Only processes can change information of G4Track and add secondary tracks via
    ParticleChange. G4VProcess is a base class of all processes and it has 3 kinds
    of DoIt and GetPhysicalInteraction methods in order to describe interactions
    generically. If a user want to modify information of G4Track, he (or she)
    SHOULD create a special process for the purpose and register the process to the
    particle.



