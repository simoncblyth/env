
 """
    
       THIS RESTRICTS THE TYPE OF TESTS THAT CAN BE DONE IN ONE GO
    
       In [3]: g.TopAlg
       Out[3]: 
['GtGenerator/Generator',
 'GtHepMCDumper/Dumper',
 'ConsistencyAlg',
 'GtGenerator/Generator',
 'GtHepMCDumper/Dumper']
 
      Traceback (most recent call last):
  File "test_consistency.py", line 170, in <module>
    conf = _configure()
  File "test_consistency.py", line 72, in _configure
    return GenToolsTestConfig( volume="/dd/Geometry/Pool/lvFarPoolIWS" )
  File "test_consistency.py", line 47, in __init__
    self.conf = gentools.GenToolsConfig(**atts)
  File "/disk/d3/dayabay/local/dyb/trunk_dbg/NuWa-trunk/dybgaudi/InstallArea/python/gentools.py", line 18, in __init__
    self.init()
  File "/disk/d3/dayabay/local/dyb/trunk_dbg/NuWa-trunk/dybgaudi/InstallArea/python/gentools.py", line 23, in init
    self.init_generator()
  File "/disk/d3/dayabay/local/dyb/trunk_dbg/NuWa-trunk/dybgaudi/InstallArea/python/gentools.py", line 82, in init_generator
    self.app.TopAlg += [ "GtGenerator/Generator", "GtHepMCDumper/Dumper" ]
  File "/disk/d3/dayabay/local/dyb/trunk_dbg/NuWa-trunk/gaudi/InstallArea/python/GaudiPython/Bindings.py", line 173, in __setattr__
    if prop.fromString( value ).isFailure() :
SystemError: problem in C++; program state has been reset
 
 
 
"""





import gtconfig
conf = gtconfig.GenToolsTestConfig( volume="/dd/Geometry/Pool/lvFarPoolIWS" )

g.run(conf.nevents())
    
#g.TopAlg -= [ "GtGenerator/Generator", "GtHepMCDumper/Dumper" , "ConsistencyAlg" ]    
g.exit()
 
conf = _configure()    ## this now just returns the singleton ... accepting the cleanup issue
g.run(conf.nevents())
    
g.exit()






