G4DAEView geometry cache 
==========================

Without the geocache (but with bvh cache as normal) takes ~13 seconds to be ready to propagate::

    delta:chroma blyth$ g4daechroma.sh 
    2014-11-27 17:19:10,541 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:62  ***** g4daechroma start
    ...
    2014-11-27 17:19:22,980 INFO    env.geant4.geometry.collada.g4daeview.daechromacontext:114 *** first GPU hit : done 
    2014-11-27 17:19:22,981 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:91  ***** post G4DAEChroma ctor
    2014-11-27 17:19:22,981 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:44  start polling responder: DAEDirectResponder connect tcp://127.0.0.1:5002  
    2014-11-27 17:19:22,981 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:48  polling 0 
    2014-11-27 17:19:34,001 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:48  polling 10 

With geocache g4daechroma (and bvh cache) is ready for propagating in under a second from launch:: 

    delta:chroma blyth$ g4daechroma.sh --geocache
    2014-11-27 17:14:28,103 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:62  ***** post DAEDirectConfig.parse
    2014-11-27 17:14:28,103 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:69  load chroma_geometry from /tmp/env/chroma_geometry 
    ...
    2014-11-27 17:14:28,701 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:91  ***** post G4DAEChroma ctor
    2014-11-27 17:14:28,701 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:44  start polling responder: DAEDirectResponder connect tcp://127.0.0.1:5002  
    2014-11-27 17:14:28,701 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:48  polling 0 
    2014-11-27 17:14:39,726 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:48  polling 10 


With geocache, getting reproducible digests::

      delta:~ blyth$ mocknuwa.sh MOCK 

        "caller":   {
            "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
            "dhitlist": "3382b4abb870dc3c5215502bf5052448",
            "dhits":    "d520df2bfe622ccabaa3af843fc07d67",
            "dphotons": "65851b617842abf44939f206b942c7b7",
            "empty":    0,
            "htag": "aaMOCK"
        }

But they are different from the non geocache digests::

        "caller":   {
            "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
            "dhitlist": "7b7d251bac9528612a6f8aebc8a61f27",
            "dhits":    "5313b547ea381eadf14983c40855face",
            "dphotons": "65851b617842abf44939f206b942c7b7",
            "empty":    0,
            "htag": "hhMOCK"
        }



Huh numerical diffs but they hit same PMTs::

    1: a = ph("aaMOCK")
    2: h = ph("hhMOCK")

    In [36]: np.all(a.pmtid == h.pmtid)
    Out[36]: DAEPhotonsNPL(True, dtype=bool)


::

    In [39]: d = a - h

    In [47]: z = np.zeros( (4,4), dtype=a.dtype )

    In [51]: for i in range(len(d)):
       ....:     if not np.allclose(d[i],z):print i,d[i]
       ....:     
    29 [[ 0.     0.     0.     0.   ]
     [ 0.802 -0.032  0.162  0.   ]
     [-1.289  1.471  0.141  0.   ]
     [ 0.     0.     0.     0.   ]]
    54 [[-1122.146   742.875   132.865     6.074]
     [   -0.591     0.485    -0.837     0.   ]
     [   -0.249     0.758     0.394     0.   ]
     [    0.        0.        0.        0.   ]]
    65 [[ 1231.623   353.375  2412.509     4.237]
     [   -1.186    -0.438    -0.325     0.   ]
     [   -0.489     1.203     0.        0.   ]
     [    0.        0.        0.        0.   ]]

::

    In [53]: count = 0 

    In [54]: for i in range(len(d)):
       ....:     if not np.allclose(d[i],z):count+=1
       ....:     

    In [55]: count
    Out[55]: 120

    In [56]: len(d)
    Out[56]: 684

    In [57]: 120./684.
    Out[57]: 0.17543859649122806




Hmm the majority are identical, 17% are discrepant, all that hit PMTs are identical::

    In [59]: a[a.pmtid > 0].shape
    Out[59]: (72, 4, 4)

    In [60]: h[h.pmtid > 0].shape
    Out[60]: (72, 4, 4)

    In [62]: np.all(a[a.pmtid > 0] == h[h.pmtid > 0])
    Out[62]: DAEPhotonsNPL(True, dtype=bool)


Probably some surface or material index in the cache differs ? 
And all photons that interact with it go awry.

::

    In [67]: len(np.where(a.history != h.history)[0])
    Out[67]: 117

    In [68]: len(np.where(a.history == h.history)[0])
    Out[68]: 567



::

    2014-11-27 17:56:26,147 INFO    env.geant4.geometry.collada.g4daeview.g4daechroma:79  ***** post make_chroma_geometry 
    2014-11-27 17:56:26,147 INFO    chroma.npycacheable :248 skip bvh type <class 'chroma.bvh.bvh.BVH'>  <chroma.bvh.bvh.BVH object at 0x11f186d50> 
    2014-11-27 17:56:26,147 INFO    chroma.npycacheable :110 save_list /tmp/env/chroma_geometry/chroma.detector:Detector:0x11aace190/charge_cdf len 2 
    2014-11-27 17:56:26,230 WARNING chroma.npycacheable :107 skip list/tuple thats too long 9068 /tmp/env/chroma_geometry/chroma.detector:Detector:0x11aace190/solid_displacements 
    2014-11-27 17:56:26,238 WARNING chroma.npycacheable :107 skip list/tuple thats too long 9068 /tmp/env/chroma_geometry/chroma.detector:Detector:0x11aace190/solid_rotations 
    2014-11-27 17:56:26,238 WARNING chroma.npycacheable :107 skip list/tuple thats too long 9068 /tmp/env/chroma_geometry/chroma.detector:Detector:0x11aace190/solids 
    2014-11-27 17:56:26,246 INFO    chroma.npycacheable :110 save_list /tmp/env/chroma_geometry/chroma.detector:Detector:0x11aace190/time_cdf len 2 
    2014-11-27 17:56:26,247 INFO    chroma.npycacheable :110 save_list /tmp/env/chroma_geometry/chroma.detector:Detector:0x11aace190/unique_materials len 29 
    2014-11-27 17:56:26,270 INFO    chroma.npycacheable :110 save_list /tmp/env/chroma_geometry/chroma.detector:Detector:0x11aace190/unique_surfaces len 31 
    2014-11-27 17:56:26,270 WARNING chroma.npycacheable :119 skipping sequence element 0 None 


::

    In [2]: chroma_geometry.unique_surfaces
    Out[2]: 
    [None,
     <Surface __dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface>,
     <Surface __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface>,
     <Surface __dd__Geometry__PoolDetails__PoolSurfacesAll__LegInIWSTubSurface>,
     ...


Hope its that None::

    114         for i in range(len(geometry.unique_surfaces)):
    115             surface = geometry.unique_surfaces[i]
    116 
    117             if surface is None:
    118                 # need something to copy to the surface array struct
    119                 # that is the same size as a 64-bit pointer.
    120                 # this pointer will never be used by the simulation.
    121                 self.surface_ptrs.append(np.uint64(0))
    122                 continue
    123 


Yep, with geocache after fixing the persisting of the None surface are back to identical results::

        "caller":   {
            "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
            "dhitlist": "7b7d251bac9528612a6f8aebc8a61f27",
            "dhits":    "5313b547ea381eadf14983c40855face",
            "dphotons": "65851b617842abf44939f206b942c7b7",
            "empty":    0,
            "htag": "bbMOCK"

        "caller":   {
            "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
            "dhitlist": "7b7d251bac9528612a6f8aebc8a61f27",
            "dhits":    "5313b547ea381eadf14983c40855face",
            "dphotons": "65851b617842abf44939f206b942c7b7",
            "empty":    0,
            "htag": "ccMOCK"
        }



Unwanted Geant4 import coming in with some chroma import with g4daechroma?::

    /usr/local/env/chroma_env/lib/python2.7/site-packages/Geant4/__init__.pyc in _run_abort(signum, frame)
        240     gRunManager.AbortRun(True)
        241   else:
    --> 242     raise KeyboardInterrupt
        243 
        244 if (threading.activeCount() == 1):



Trying with g4daeview.py have diverged again::


    "caller":   {
        "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
        "dhitlist": "1af89ad33e6eaeba73da0ba62f19214f",
        "dhits":    "34b666d4225350ae22e1643ce94278fe",
        "dphotons": "65851b617842abf44939f206b942c7b7",
        "empty":    0,
        "htag": "ddMOCK"
    }


Hmm here g4daeview without the geocache, get a match to with geocache::

    "caller":   {
        "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
        "dhitlist": "1af89ad33e6eaeba73da0ba62f19214f",
        "dhits":    "34b666d4225350ae22e1643ce94278fe",
        "dphotons": "65851b617842abf44939f206b942c7b7",
        "empty":    0,
        "htag": "ffMOCK"


::

    In [86]: np.all(c == h)
    Out[86]: DAEPhotonsNPL(True, dtype=bool)

    b = ph("ddMOCK")

    In [80]: b[b.pmtid>0].shape
    Out[80]: (76, 4, 4)

    In [81]: a[a.pmtid>0].shape
    Out[81]: (72, 4, 4)


Most all are different::

    In [92]: count = 0 

    In [93]: for i in range(len(d)):
       ....:     if not np.allclose(d[i],z):count+=1
       ....:     

    In [94]: count
    Out[94]: 681





With a fresh updated cache::

    delta:chroma blyth$ g4daechroma.sh --geocacheupdate

    "caller":   {
        "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
        "dhitlist": "7b7d251bac9528612a6f8aebc8a61f27",
        "dhits":    "5313b547ea381eadf14983c40855face",
        "dphotons": "65851b617842abf44939f206b942c7b7",
        "empty":    0,
        "htag": "gdcMOCK"
    }

Operating from that cache get agreement::

    delta:chroma blyth$ g4daechroma.sh --geocache

    "caller":   {
        "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
        "dhitlist": "7b7d251bac9528612a6f8aebc8a61f27",
        "dhits":    "5313b547ea381eadf14983c40855face",
        "dphotons": "65851b617842abf44939f206b942c7b7",
        "empty":    0,
        "htag": "gdccMOCK"
    }


Running off that cache::

    delta:chroma blyth$ g4daeview.sh --geocache

Get a bookmark related need for solids::

    Traceback (most recent call last):
      File "/Users/blyth/env/bin/g4daeview.py", line 4, in <module>
        main()
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/g4daeview.py", line 352, in main
        scene = DAEScene(geometry, chroma_geometry, config )
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daescene.py", line 74, in __init__
        self.bookmarks = DAEBookmarks(config, geometry ) 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daebookmarks.py", line 42, in __init__
        self.load(ipath, geometry)  
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daebookmarks.py", line 176, in load
        view = DAEViewpoint.fromini( cfg, geometry ) 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daeviewpoint.py", line 347, in fromini
        solid = geometry.find_solid_by_index(v) 
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/env/geant4/geometry/collada/g4daeview/daegeometry.py", line 390, in find_solid_by_index
        selection = filter(lambda _:str(_.index) == index, self.solids)
    AttributeError: 'DAEGeometry' object has no attribute 'solids'


After avoiding that succeed to launch in about a second::

    delta:chroma blyth$ g4daeview.sh --geocache

But still mismatched::

    "caller":   {
        "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
        "dhitlist": "1af89ad33e6eaeba73da0ba62f19214f",
        "dhits":    "34b666d4225350ae22e1643ce94278fe",
        "dphotons": "65851b617842abf44939f206b942c7b7",
        "empty":    0,
        "htag": "gdvMOCK"
    }



::

    v = ph("gdvMOCK")
    c = ph("gdcMOCK")


    In [17]: z = np.zeros((4,4), dtype=c.dtype)

    In [18]: cmv = c - v

    In [19]: count = 0 

    In [20]: for i in range(len(cmv)):
       ....:     if not np.allclose(cmv[i],z):count += 1
       ....:     

    In [21]: count
    Out[21]: 681

    In [22]: len(cmv)
    Out[22]: 684



Hmm only first last and middle agree, thats suspicious::

    In [27]: for i in range(len(cmv)):
       ....:     if np.allclose(cmv[i],z):print i 
       ....:     
    0
    341
    683

::

    In [28]: 684/2
    Out[28]: 342



::

    n [36]: c[0,3].view(np.int32)
    Out[36]: DAEPhotonsNPL([0, 0, 2, 0], dtype=int32)

    In [37]: c[341,3].view(np.int32)
    Out[37]: DAEPhotonsNPL([     341,        0,        4, 16909842], dtype=int32)

    In [38]: c[683,3].view(np.int32)
    Out[38]: DAEPhotonsNPL([683,   0,  64,   0], dtype=int32)




Fixed Problem was the time sorting again
-----------------------------------------

* need to reduce duplication between g4daeview and g4daechroma some more
  to avoid such issues


::

    (chroma_env)delta:g4daeview blyth$ g4daechroma.sh --geocache

    "caller":   {
        "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
        "dhitlist": "1af89ad33e6eaeba73da0ba62f19214f",
        "dhits":    "34b666d4225350ae22e1643ce94278fe",
        "dphotons": "65851b617842abf44939f206b942c7b7",
        "empty":    0,
        "htag": "gdcMOCK"
    }

    "caller":   {
        "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
        "dhitlist": "1af89ad33e6eaeba73da0ba62f19214f",
        "dhits":    "34b666d4225350ae22e1643ce94278fe",
        "dphotons": "65851b617842abf44939f206b942c7b7",
        "empty":    0,
        "htag": "gdcnocacheMOCK"
    }




::

    (chroma_env)delta:g4daeview blyth$ g4daeview.sh --geocache


    "caller":   {
        "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
        "dhitlist": "1af89ad33e6eaeba73da0ba62f19214f",
        "dhits":    "34b666d4225350ae22e1643ce94278fe",
        "dphotons": "65851b617842abf44939f206b942c7b7",
        "empty":    0,
        "htag": "gdvMOCK"
    }

    "caller":   {
        "COLUMNS":  "htag:s,dphotons:s,dhits:s,dhitlist:s",
        "dhitlist": "1af89ad33e6eaeba73da0ba62f19214f",
        "dhits":    "34b666d4225350ae22e1643ce94278fe",
        "dphotons": "65851b617842abf44939f206b942c7b7",
        "empty":    0,
        "htag": "gdvnocacheMOCK"
    }






