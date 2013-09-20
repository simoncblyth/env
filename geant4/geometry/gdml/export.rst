Export GDML 
=============

* :google:`geant4 giga gdml`
* http://svn.cern.ch/guest/lhcb/Gauss/trunk/Sim/LbGDML/options/GDMLWriter.opts

::

    GiGa.RunSeq.Members += { "GDMLRunAction"};
    // The following two lines have the default
    GiGa.RunSeq.GDMLRunAction.Schema = "$G4GDMLROOT/schema/gdml.xsd";
    GiGa.RunSeq.GDMLRunAction.Output = "LHCb.gdml";

