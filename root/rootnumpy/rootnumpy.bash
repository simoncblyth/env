# === func-gen- : root/rootnumpy/rootnumpy fgp root/rootnumpy/rootnumpy.bash fgn rootnumpy fgh root/rootnumpy
rootnumpy-src(){      echo root/rootnumpy/rootnumpy.bash ; }
rootnumpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rootnumpy-src)} ; }
rootnumpy-vi(){       vi $(rootnumpy-source) ; }
rootnumpy-env(){      elocal- ; }
rootnumpy-usage(){ cat << EOU

* http://rootpy.github.io/root_numpy/install.html

http://scikit-hep.org/root_numpy/install.html

::

   [blyth@localhost root_numpy]$ ROOTSYS=$ROOTSYS ~/anaconda2/bin/python setup.py install --user



    [blyth@localhost junotop]$ ~/anaconda2/bin/ipython

    In [2]: from root_numpy import root2array

    In [3]: a = root2array("./data/Simulation/ElecSim/PmtData.root")

    In [4]: a
    Out[4]: 
    array([(     0, 0, 0.70001458, 1.06171406, 0.34451 , 0.02934785, 0.00979265, 19943.8 , 17.99107433, 42.56181153, 19365., 2.77,  32.57),
           (     1, 1, 0.6651327 , 0.996496  , 0.26851 , 0.0963965 , 0.00038277,  8788.97,  2.70120107, 38.72334808, 19365., 2.77,  84.  ),
           (     2, 0, 0.70002625, 1.08594796, 0.323042, 0.0285224 , 0.00671892, 30365.8 , 17.57132939, 36.70343637, 19365., 2.77, 135.43),
           ...,
           (325597, 0, 0.24868   , 0.        , 0.      , 0.        , 0.        ,   119.  ,  0.        ,  0.        ,     0., 0.  ,   0.  ),
           (325598, 0, 0.248741  , 0.        , 0.      , 0.        , 0.        ,  1406.  ,  0.        ,  0.        ,     0., 0.  ,   0.  ),
           (325599, 0, 0.258522  , 0.        , 0.      , 0.        , 0.        ,  1025.  ,  0.        ,  0.        ,     0., 0.  ,   0.  )],
          dtype=[('pmtId', '<i4'), ('isHamamatsu', '<i4'), ('efficiency', '<f8'), ('gain', '<f8'), ('sigmaGain', '<f8'), ('afterPulseProb', '<f8'), ('prePulseProb', '<f8'), ('darkRate', '<f8'), ('timeSpread', '<f8'), ('timeOffset', '<f8'), ('pmtPosX', '<f8'), ('pmtPosY', '<f8'), ('pmtPosZ', '<f8')])

    In [5]: 


    In [5]: a.shape
    Out[5]: (43213,)



Need JUNOEnv for the ROOT, but then get py27 ?::

    [blyth@localhost junotop]$ ./pmtdata_Lpmt.py 
    Traceback (most recent call last):
      File "./pmtdata_Lpmt.py", line 4, in <module>
        from root_numpy import root2array
      File "/home/blyth/.local/lib/python2.7/site-packages/root_numpy/__init__.py", line 22, in <module>
        config = get_config()
      File "/home/blyth/.local/lib/python2.7/site-packages/root_numpy/setup_utils.py", line 117, in get_config
        from pkg_resources import resource_filename
      File "/home/blyth/junotop/ExternalLibs/ROOT/6.20.02/lib/ROOT.py", line 522, in _importhook
        return _orig_ihook( name, *args, **kwds )
    ImportError: No module named pkg_resources
    [blyth@localhost junotop]$ 



EOU
}
rootnumpy-dir(){ echo $(local-base)/env/root/root_numpy ; }
rootnumpy-cd(){  cd $(rootnumpy-dir); }
rootnumpy-get(){
   local dir=$(dirname $(rootnumpy-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/rootpy/root_numpy.git
}
