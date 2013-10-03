GDML Approaches Explored
=========================

`export.py`
        NuWa module that uses `GiGaRunActionGDML` to get NuWa to export `.gdml` via GiGa
`export.sh`
        Bash `nuwa.py` invokation of `export.py`
`gdml_traverse.py`
        Cursory GDML parse and creation of SQLite DB. Objective was to make a comparison with 
        the shapedb created from WRL exported via VRML2FILE. Not pursued. 
`gdml.py`
        Full GDML XML parse using Elementtree, with creation of walkable volume hierarchy
`g4gdml.py`
        Use of Geant4Py (a `boost_python` wrapping of Geant4) to use the G4GDMLParser from python, 
        giving access to the real G4 volume hierarchy.
        Unfortunately the wrapping misses the vital `G4VSolid::GetPolyhedron`.  
        MAYBE: Try adding that to G4Py
`gdmltest.cc`
       Geant4 main that uses G4GDMLParser to read the .gdml and spit out vertices.  Did not pursue
       as too much reinventing wheel. Instead proceeded to the VRML2FILE and GDML export codes as starting points.



