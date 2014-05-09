NuWa CMT Debugging
====================

CMT refs : for rootcint/reflex woes 
--------------------------------------------

* https://savannah.cern.ch/bugs/index.php?52581
* http://root.cern.ch/root/roottalk/roottalk09/1130.html
* http://root.cern.ch/root/roottalk/roottalk10/1259.html
* http://lhcb-comp.web.cern.ch/lhcb-comp/support/CMT/CMT_SCRAM_21.htm
* http://cms02.phys.ntu.edu.tw/tracs/env/browser/trunk/eve/SplitGLView/cmt/requirements?rev=1612
* https://wiki.bnl.gov/dayabay/index.php?title=Component_and_Linker_Libraries
* http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/562
* http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/771 DBI from rootcint to reflex
* https://wiki.bnl.gov/dayabay/index.php?title=GaudiObjDesc  tip about linking to generated Dict libs


CMT Optional Dependencies, env setup
--------------------------------------

* https://wiki.bnl.gov/dayabay/index.php?title=Installing_Optional_External_Packages
* https://wiki.bnl.gov/dayabay/index.php?title=CMTHOME_and_CMTUSERCONTEXT



Fragment Analysis
-------------------



* https://halldweb1.jlab.org/wiki/index.php/Serializing_and_deserializing_root_objects

::

    [blyth@belle7 dybx]$ find . -name fragments
    ./NuWa-trunk/lcgcmt/LCG_Policy/cmt/fragments
    ./NuWa-trunk/lcgcmt/LCG_Interfaces/Qt/cmt/fragments
    ./NuWa-trunk/lcgcmt/LCG_Interfaces/SEAL/cmt/fragments
    ./NuWa-trunk/lcgcmt/LCG_Interfaces/Boost/cmt/fragments
    ./NuWa-trunk/lcgcmt/LCG_Interfaces/ROOT/cmt/fragments
    ./NuWa-trunk/lhcb/Kernel/LHCbKernel/cmt/fragments
    ./NuWa-trunk/lhcb/GaudiObjDesc/fragments
    ./NuWa-trunk/lhcb/Vis/OSCONX/cmt/fragments
    ./NuWa-trunk/dybgaudi/Detector/XmlDetDesc/cmt/fragments
    ./NuWa-trunk/dybgaudi/Detector/MiniDryRunXmlDetDesc/cmt/fragments
    ./NuWa-trunk/dybgaudi/Utilities/GenDict/fragments
    ./NuWa-trunk/dybgaudi/Utilities/Hephaestus/cmt/fragments
    ./NuWa-trunk/dybgaudi/Utilities/FillTmpl/fragments
    ./NuWa-trunk/dybgaudi/Documentation/OfflineUserManual/cmt/fragments
    ./NuWa-trunk/dybgaudi/DybPolicy/cmt/fragments
    ./NuWa-trunk/gaudi/GaudiObjDesc/fragments
    ./NuWa-trunk/gaudi/GaudiPolicy/cmt/fragments
    ./external/build/LCG/OpenScientist/Interfaces/ROOT/v1r51000p0/cmt/fragments
    ./external/CMT/v1r20p20080222/src/demo/demoA/fragments
    ./external/CMT/v1r20p20080222/mgr/fragments
    [blyth@belle7 dybx]$ 
     

Notable fragments::

    [blyth@belle7 dybx]$ l NuWa-trunk/dybgaudi/DybPolicy/cmt/fragments/
    total 24
    -rw-rw-r-- 1 blyth blyth  476 Mar  5 20:28 rootcint_dictionary
    -rw-rw-r-- 1 blyth blyth  140 Mar  5 20:28 rootcint_dictionary_header
    -rw-rw-r-- 1 blyth blyth  210 Mar  5 20:28 rootcint_dictionary_trailer
    -rw-rw-r-- 1 blyth blyth 1112 Mar  5 20:28 rootmap
    -rw-rw-r-- 1 blyth blyth    1 Mar  5 20:28 rootmap_header
    -rw-rw-r-- 1 blyth blyth    1 Mar  5 20:28 rootmap_trailer
    [blyth@belle7 dybx]$ 

    NuWa-trunk/lcgcmt/LCG_Policy/cmt/fragments/lcg_dictionary_generator


::

    [blyth@belle7 dybx]$ l NuWa-trunk/lcgcmt/LCG_Interfaces/ROOT/cmt/fragments/
    total 28
    -rw-rw-r-- 1 blyth blyth    0 Mar  5 20:27 genmap
    -rw-rw-r-- 1 blyth blyth  627 Mar  5 20:27 genmap_header
    drwxrwxr-x 3 blyth blyth 4096 Mar  5 20:27 nmake
    -rwxrwxr-x 1 blyth blyth 1906 Mar  5 20:27 reflex_dictionary_generator
    -rwxrwxr-x 1 blyth blyth    0 Mar  5 20:27 reflex_dictionary_generator_header
    -rwxrwxr-x 1 blyth blyth   52 Mar  5 20:27 reflex_dictionary_generator_trailer
    -rwxrwxr-x 1 blyth blyth  248 Mar  5 20:27 rootcint
    -rwxrwxr-x 1 blyth blyth  131 Mar  5 20:27 rootcint_header
    -rwxrwxr-x 1 blyth blyth  195 Mar  5 20:27 rootcint_trailer


* NuWa-trunk/gaudi/GaudiKernel/cmt/requirements

* NuWa-trunk/lcgcmt/LCG_Interfaces/Reflex/cmt/requirements

::

     34 pattern reflex_dictionary \
     35   private ;\
     36   document reflex_dictionary_generator <dictionary>Gen <headerfiles> dictionary=<dictionary> libdirname=lib ; \
     37   library <dictionary>Dict -import=<imports> -import=Reflex -s=../$(tag)/dict/<dictionary> *.cpp ; \
     38   macro <dictionary>_reflex_selection_file " <selectionfile> " ; \
     39   macro <dictionary>_rootmap       $(<PACKAGE>ROOT)/$(tag)/<dictionary>Dict.rootmap \
     40         <project>_with_installarea $(CMTINSTALLAREA)/$(tag)/lib/<project>Dict.rootmap ;\
     41   macro <dictionary>_reflex_options " <options> $(gccxmlopts) --select=<selectionfile>  --gccxmlpath=$(GCCXML_home)/bin" ; \
     42   macro <dictionary>Dict_dependencies "$(<package>_linker_library) <dictionary>Gen" ;\
     43   macro <dictionary>Dict_shlibflags "$(libraryshr_linkopts) $(cmt_installarea_linkopts) $(<package>_linkopts) $(<dictionary>Dict_use_linkopts) " ;\
     44   end_private
     45 



* NuWa-trunk/dybgaudi/DybPolicy/cmt/requirements 

::

      1 package DybPolicy
      2 version v0
      3 
      4 apply_tag dayabay
      5 
      6 use GaudiPolicy  *
      7 



* NuWa-trunk/gaudi/GaudiPolicy/cmt/requirements

::

     094 pattern library_Llinkopts \
     095     macro_append <package>_linkopts                 "" \
     096          <project>_with_installarea&Unix     " -l<library> "\
     097          <project>_with_installarea&WIN32    " <library>.lib "\
     098          <project>_without_installarea&Unix  " -L$(<package>Dir) -l<library> "\
     099          <project>_without_installarea&WIN32 " $(<package>Dir)/<library>.lib "\
     100                                 Unix&static  " $(<package>Dir)/lib<library>.a "
     101 
     102 
     103 pattern library_Lshlibflags \
     104     private ; \
     105     macro <library>_shlibflags "$(libraryshr_linkopts) $(cmt_installarea_linkopts) $(<library>_use_linkopts)" ; \
     106     macro_remove <library>_use_linkopts "$(<package>_linkopts)" ; \
     107     end_private
     108 
     109 pattern library_Softlinks \
     110     macro_append <package>_libraries        "" \
     111          <project>_without_installarea&Unix " <library> "\
     112                  WIN32                      "" \
     113                  Unix&static                ""
     ...
     118 pattern library_path \
     119     path_remove PATH ""  WIN32 "\<package>\" ; \
     120     path_prepend PATH ""  \
     121         <project>_without_installarea&WIN32 "${<package>_root}\${<package>_tag}" ; \
     122     path_remove DYLD_LIBRARY_PATH "" Darwin "/<package>/" ; \
     123     path_append DYLD_LIBRARY_PATH "" \
     124         <project>_without_installarea&Darwin "${<package>_root}/${<package>_tag}" ; \
     125     apply_pattern library_Softlinks library="<library>"
     126 
     127 
     128 pattern component_library \
     129     apply_pattern libraryShr library="<library>" ; \
     130     apply_pattern library_Clinkopts library="<library>" ; \
     131     apply_pattern library_Cshlibflags library="<library>" ;\
     132     macro <library>_dependencies "$(<package>_linker_library) " ;\
     133     apply_pattern generate_rootmap library=<library> group=<group> ; \
     134     apply_pattern generate_configurables library=<library> group=<group>
     135 
     136 pattern linker_library \
     137     apply_pattern library_path        library="<library>" ; \
     138     apply_pattern library_Llinkopts   library="<library>" ; \
     139     apply_pattern library_Lshlibflags library="<library>" ; \
     140     apply_pattern library_stamps      library="<library>" ; \
     141     macro <package>_linker_library    "<library>"
     142 




Code In Question
------------------

#. ZMQRoot,  simple SendObject/ReceiveObject API using TObject, but does not inherit from TObject
#. MyTMessage, inherits from TMessage, has `ClassDef`, crucially needs introspection  






CMT : Dict not being generated first 
----------------------------------------------


::

    blyth@belle7 cmt]$ cmt show pattern reflex_dictionary          
    # Reflex v1 defines pattern reflex_dictionary as
      private 
       document reflex_dictionary_generator "<dictionary>Gen" "<headerfiles>" "dictionary=<dictionary>" "libdirname=lib" 
       library <dictionary>Dict "-import=<imports>" "-import=Reflex" "-s=../$(tag)/dict/<dictionary>" "*.cpp" 
       macro <dictionary>_reflex_selection_file " <selectionfile> " 
       macro <dictionary>_rootmap "$(<PACKAGE>ROOT)/$(tag)/<dictionary>Dict.rootmap" "<project>_with_installarea" "$(CMTINSTALLAREA)/$(tag)/lib/<project>Dict.rootmap" 
       macro <dictionary>_reflex_options " <options> $(gccxmlopts) --select=<selectionfile>  --gccxmlpath=$(GCCXML_home)/bin" 
       macro <dictionary>Dict_dependencies "$(<package>_linker_library) <dictionary>Gen" 
       macro <dictionary>Dict_shlibflags "$(libraryshr_linkopts) $(cmt_installarea_linkopts) $(<package>_linkopts) $(<dictionary>Dict_use_linkopts) " 
       end_private
    # ZMQRoot v1 applies pattern reflex_dictionary => 
    private 
       document reflex_dictionary_generator ZMQRootGen ${ZMQROOTROOT}/dict/headers.h dictionary=ZMQRoot libdirname=lib 
       library ZMQRootDict -import= -import=Reflex -s=../$(tag)/dict/ZMQRoot *.cpp 
       macro ZMQRoot_reflex_selection_file  ${ZMQROOTROOT}/dict/classes.xml  
       macro ZMQRoot_rootmap $(ZMQROOTROOT)/$(tag)/ZMQRootDict.rootmap dybgaudi_with_installarea $(CMTINSTALLAREA)/$(tag)/lib/dybgaudiDict.rootmap 
       macro ZMQRoot_reflex_options   $(gccxmlopts) --select=${ZMQROOTROOT}/dict/classes.xml  --gccxmlpath=$(GCCXML_home)/bin 
       macro ZMQRootDict_dependencies $(ZMQRoot_linker_library) ZMQRootGen 
       macro ZMQRootDict_shlibflags $(libraryshr_linkopts) $(cmt_installarea_linkopts) $(ZMQRoot_linkopts) $(ZMQRootDict_use_linkopts)  
       end_private
    [blyth@belle7 cmt]$ 



::

 g++ \
     -L/data1/env/local/dybx/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib \
     -L/data1/env/local/dybx/NuWa-trunk/gaudi/InstallArea/i686-slc5-gcc41-dbg/lib \
     -L/data1/env/local/dybx/NuWa-trunk/lhcb/InstallArea/i686-slc5-gcc41-dbg/lib  \
      -L/data1/env/local/dybx/NuWa-trunk/relax/InstallArea/i686-slc5-gcc41-dbg/lib  \
           -shared -m32 -o libZMQRootDict.so \
          "headers_dict.o" -fPIC -ldl -Wl,--no-undefined -m32 \ 
     -L/data1/env/local/dybx/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib \
      -L/data1/env/local/dybx/NuWa-trunk/gaudi/InstallArea/i686-slc5-gcc41-dbg/lib \
      -L/data1/env/local/dybx/NuWa-trunk/lhcb/InstallArea/i686-slc5-gcc41-dbg/lib \
      -L/data1/env/local/dybx/NuWa-trunk/relax/InstallArea/i686-slc5-gcc41-dbg/lib \
      -L/data1/env/local/dybx/NuWa-trunk/../external/ROOT/5.26.00e_python2.7/i686-slc5-gcc41-dbg/root/lib -lReflex \
      -L/data1/env/local/dybx/NuWa-trunk/../external/ROOT/5.26.00e_python2.7/i686-slc5-gcc41-dbg/root/lib -lCore -lCint -lTree -lpthread -lRIO -lNet -lrt -lGpad -lHist -lPhysics -lGeom -lGraf -lGenVector -lMathCore -lNet \
       -L/data1/env/local/dybx/NuWa-trunk/../external/ROOT/5.26.00e_python2.7/i686-slc5-gcc41-dbg/root/lib -lCore -lCint -lTree -lpthread -lRIO -lNet -lrt \
        -L/data1/env/local/dybx/NuWa-trunk/../external/zmq/4.0.4/i686-slc5-gcc41-dbg/lib -lzmq





::

    [blyth@belle7 cmt]$ cmt show pattern rootcint_dictionary
    # DybPolicy v0 defines pattern rootcint_dictionary as
      public 
       apply_pattern ld_library_path 
       apply_pattern public_package_include 
       private 
       apply_pattern private_package_include 
       macro <package>_rootmap "$(<PACKAGE>ROOT)/$(tag)/$(rootmap_name)" "<project>_with_installarea" "$(CMTINSTALLAREA)/$(tag)/lib/$(rootmap_name)" 
       macro_append <package>_rootcint_source "../$(tag)/rootcint/*Dict.cc" 
       macro rootcint_dict_suffix "Dict" 
       macro <package>_rootcint_linkdef_file "<linkdeffile>" 
       document rootcint_dictionary "<package>_cint" "<headerfiles>" 
       document rootmap "<package>_rootmap" 
       public 

    [blyth@belle7 cmt]$ cmt show fragment rootcint_dictionary
    ${DYBPOLICYROOT}/cmt/fragments/rootcint_dictionary

    [blyth@belle7 cmt]$ cat ${DYBPOLICYROOT}/cmt/fragments/rootcint_dictionary
    $(dictdir)/${CONSTITUENT}.rootcint_dictionary : $(dictdir)/${CONSTITUENT}Dict.cpp
        @echo $@

    gensrcdict=$(dictdir)/${CONSTITUENT}Dict.cc


    ${CONSTITUENT} ::  $(gensrcdict)
        @:


    $(gensrcdict) : ${FULLNAME}
        @echo Generating ROOT Dictionary $@ 
        @-mkdir -p $(dictdir) 
        $(rootcint) -f $(dictdir)/${CONSTITUENT}Dict.cc -c \
                       ${${package}_cintflags} $(pp_cppflags) \
               $(includes) $(use_pp_cppflags) ${FULLNAME} \
               $(${package}_rootcint_linkdef_file)



    [blyth@belle7 cmt]$ 




