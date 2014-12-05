G4DAEChroma Memory Leak
=========================

Running a mocknuwa scan matrix of 16(configs)x48(batches) 
reveals the python g4daechroma.py 
propagating process to balloon in VSIZE shown by top.
Got to 30G and the machine became sluggish before interrupting
it at config 11.

::

    mocknuwa-scan(){
       mocknuwa-runenv 
       MockNuWa 1:49 1:17
    }




