DAE TOOLS
==========

``daenode.py``
           Based on **pycollada**. Reconstructs Geant4 tree and provides subgeometry extraction.
``daedb.py``
           Based on **daenode**.  Creates summary sqlite3 DB from daenode/pycollada accessed geometry info.
``dae_cf_wrl.py``
           Comparison of geometry info in sqlite3 DBs created from VRML2 and DAE files. 
``daediff.py``
           Based on **lxml**. Compares original DAE with a from the root subcopy, validating the subcopy mechanism of daenode.
``daegeom.py``
           Dumps geometry info from DAE and WRL. The slowness of repeated parsing of the DAE file motivated development
           of the ``daeserver``.
``daestat.py``
           Basic statistics on collada files, using lxml, **not** pycollada
``xmldae.py``
           XML view of collada files, using lxml and **not** pycollada



