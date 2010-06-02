
== Overview ==

  ipython + pyROOT + Eve(OpenGL)  event display architecture  


== Usage ==

  abtviz-
  abtbiz

  OR 
  abtbiz-ipython   
  In [1]: run ev.py      ## run ev.py from ipython prompt 
  
  GUI window will popup, and the ipython session will remain active 
  allowing both to be used, note the important globals to use on the
  ipython commandline
  
      g   :  python Controller class 
      g_  :  underlying C++ EvModel
      gMQ :  message queue instance 
      
  Use tab completion on these to see methods etc and "?"       
      

== Debugging ==

  abtviz-
  abtviz-gpython

  (gdb) set args ev.py
  (gdb) run


== Source Index ==

=== Specialized Controllers, "mains" ===

  ev.py      :   Controller 
                     a specialization of the EvController
                     bringing together the Ev* classes 
                       
  evb.py     :   Controller
                     a simplified EvController specialization without the GUI 
                     primarily for debugging 

=== Primary Ev* Components ===
  
  evctrl.py   :  EvController   
                       manages connections between the underlying model (using EvConnectModel)
                       and provides default handlers for model changes, 
                       that are expected to be overridden in subclass controllers
  
  evgeom.py   :  EvGeom    
                       manages loading of the TEve detector geometry prepared by prepare_geom.py   
                       and setting of hitmaps    
  evdigi.py   :  
 
                 PMTDigi
                        visual representation of PMT response
                 EvDigi
                        list of PMTDigi providing updating 
  
  evgui.py    :  EvGui
                        adds widgets to the TEveBrowser GUI and provides handlers for 
                        button presses, menu choices that manipulate the 
                        underlying C++ EvModel via the g_ global  
  
  
  
  evtree.py   :  EvTree
                         tree handling providing pmt_response and tracker_hits 
                         with help of EvDataModel
                         
  evonline.py :  EvOnline
                         hookup to message queue event feed 


=== Support Ev* Components ===


  evconnectmodel.py   : EvConnectModel
                            ROOT signal wiring that notifies changes to 
                            the underling trivial C++ EvModel (source + cursor) 
                            up to the pyROOT EvController
                          
  evdatamodel.py      : EvDataModel 
                            focus event data model (ROOT object graph) specifics into 
                            a single class in order to act as buffer against data model changes 
                           
=== Utility Components ===  
  
  geoconf.py          : GeoConf, VolMatcher     
  geoedit.py
                          geometry specifics using volume name pattern matching to color the volumes

                         
  prepare_geom.py
                         once per installation conversion of root TGeo geometry into form 
                         needed by TEve  
                         
                         usage :
                              cd $ABERDEEN_HOME/AbtViz
                              make ipython 
                              In [1]: run prepare_geom.py
                         

  pmtmap.py
                         PMT coordinates

  


  src/
          EvModel{.cc,.h,_LinkDef.hh}
                simplest useful model for event display ... a list of events and pointer into it 
          
          KeyHandler{.cc,.h,_LinkDef.hh}
                support for shortcut keys : up/down arrows for event navigation
                
                
                
  tests/
                ? trying out different architectures for online event display ?
  
           dispatch.py
           MyDispatcher.C
           test_pmq.py
           watcher.py




 == potential issues ==
 
 
     1) Model changes requiring fixes to evdatamodel.py are the most likely issues ...
 
 
 
 
 
 


                
