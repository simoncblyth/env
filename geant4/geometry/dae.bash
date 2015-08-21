# === func-gen- : geant4/geometry/dae fgp geant4/geometry/dae.bash fgn dae fgh geant4/geometry
dae-src(){      echo geant4/geometry/dae.bash ; }
dae-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dae-src)} ; }
dae-vi(){       vi $(dae-source) ; }
dae-env(){      elocal- ; }
dae-usage(){ cat << EOU

DAE based on GDML code
========================

* http://www.khronos.org/files/collada_spec_1_4.pdf 
* https://github.com/KhronosGroup/COLLADA-CTS/blob/master/StandardDataSets/collada/library_nodes/node/1inst2inLN/1inst2inLN.dae

  * a good source of non-trivial collada examples 

The *rotate* element contains a list of four floating-point values, 
similar to rotations in the OpenGL and RenderMan specification. 
These values are organized into a column vector [X,Y,Z] 
specifying the axis of rotation followed by an angle in degrees. 

Observations
-------------

G4DAE direct exports
~~~~~~~~~~~~~~~~~~~~~~~

#. open without warning or error
#. universe often too big, after zooming in 
   it is difficult to navigate 

#. Xcode nodes are shown as "untitled"
 
   * note that the node elements have id but no name attributes


Subtrees created by g4daenode (using pycollada)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. lack xml shebang headline::

   <?xml version="1.0" encoding="ISO-8859-1"?>
 
#. fails to load into Preview, with 
#. order 11 warnings in Xcode 
#. Xcode nodes are named appropriately, node elements have name attributes 


Fixed node elements swap
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    g4daenode.sh --geom 3155___5 > 3155___5cor.dae



Xcode warnings with 3155___2.dae 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 15762, 15805, 15812, 16126, 16133


::

     warning: File Consistency Issue: Line 15762: Element '{http://www.collada.org/2005/11/COLLADASchema}matrix': 
         This element is not expected. Expected is one of ( 
               {http://www.collada.org/2005/11/COLLADASchema}instance_node, 
               {http://www.collada.org/2005/11/COLLADASchema}node, 
               {http://www.collada.org/2005/11/COLLADASchema}extra ).


::


    15754     <node name="__dd__Geometry__PMT__lvPmtHemi0xc133740" id="__dd__Geometry__PMT__lvPmtHemi0xc133740">
    15755       <instance_geometry url="#pmt-hemi0xc0fed90">
    15756         <bind_material>
    15757           <technique_common>
    15758             <instance_material symbol="Pyrex" target="#__dd__Materials__Pyrex0xc1005e0"/>
    15759           </technique_common>
    15760         </bind_material>
    15761       </instance_geometry>
    15762*      <node name="__dd__Geometry__PMT__lvPmtHemi--pvPmtHemiVacuum0xc1340e8" id="__dd__Geometry__PMT__lvPmtHemi--pvPmtHemiVacuum0xc1340e8">
                    <instance_node url="#__dd__Geometry__PMT__lvPmtHemiVacuum0xc2c7cc8"/>
                    <matrix>
    15763                 1 0 0 0
    15764                 0 1 0 0
    15765                 0 0 1 0
    15766                 0.0 0.0 0.0 1.0
    15767           </matrix>
    15768       </node>
    15769     </node>







::

    15797     <node name="__dd__Geometry__PMT__lvHeadonPmtAssy0xbf9fb20" id="__dd__Geometry__PMT__lvHeadonPmtAssy0xbf9fb20">
    15798       <instance_geometry url="#headon-pmt-assy0xbf55198">
    15799         <bind_material>
    15800           <technique_common>
    15801             <instance_material symbol="Vacuum" target="#__dd__Materials__Vacuum0xbf9fcc0"/>
    15802           </technique_common>
    15803         </bind_material>
    15804       </instance_geometry>
    15805*      <node name="__dd__Geometry__PMT__lvHeadonPmtAssy--pvHeadonPmtGlass0xc2cd968" id="__dd__Geometry__PMT__lvHeadonPmtAssy--pv      HeadonPmtGlass0xc2cd968"><instance_node url="#__dd__Geometry__PMT__lvHeadonPmtGlass0xc2c8460"/><matrix>
    15806                 1 0 0 0
    15807                 0 1 0 0
    15808                 0 0 1 0
    15809                 0.0 0.0 0.0 1.0
    15810               </matrix>
    15811         </node>
    15812*      <node name="__dd__Geometry__PMT__lvHeadonPmtAssy--pvHeadonPmtBase0xbf58520" 
                        id="__dd__Geometry__PMT__lvHeadonPmtAssy--pvHeadonPmtBase0xbf58520">
                         <instance_node url="#__dd__Geometry__PMT__lvHeadonPmtBase0xc25d120"/>
                         <matrix>
    15813                 1 0 0 0
    15814                 0 1 0 0
    15815                 0 0 1 -82.5
    15816                 0.0 0.0 0.0 1.0
    15817                </matrix>
    15818         </node>
    15819     </node>



    16126       <node name="__dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xc107a80" id="__dd__Geometry__      CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xc107a80"><instance_node url="#__dd__Geometry__CalibrationSource      s__lvWallLedDiffuserBall0xc3aa498"/><matrix>
    16127                 1 0 0 0
    16128 0 1 0 0
    16129 0 0 1 0
    16130 0.0 0.0 0.0 1.0
    16131 </matrix>

    16133*       <node name="__dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedAcrylicRod0xc28cf28" id="__dd__Geometry__Ca      librationSources__lvWallLedSourceAssy--pvWallLedAcrylicRod0xc28cf28"><instance_node url="#__dd__Geometry__CalibrationSources__l      vWallLedAcrylicRod0xc347a18"/><matrix>
    16134                 1 0 0 0
    16135 0 1 0 0
    16136 0 0 1 43.5725
    16137 0.0 0.0 0.0 1.0
    16138 </matrix>
    16139         </node>
    16140     </node>



Here in original have matrix,instance_node by sub-collada is reversed::

     72822       <node id="__dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedAcrylicRod0xc28cf28">
     72823         <matrix>
     72824                 1 0 0 0
     72825 0 1 0 0 
     72826 0 0 1 43.5725
     72827 0.0 0.0 0.0 1.0
     72828 </matrix>     
     72829         <instance_node url="#__dd__Geometry__CalibrationSources__lvWallLedAcrylicRod0xc347a18"/>
     72830         <extra>
     72831           <meta id="/dd/Geometry/CalibrationSources/lvWallLedSourceAssy#pvWallLedAcrylicRod0xc28cf28">
     72832             <copyNo>1001</copyNo>
     72833             <ModuleName></ModuleName>
     72834           </meta>
     72835         </extra>
     72836       </node>
     72837     </node>



pycollada debug
~~~~~~~~~~~~~~~~~

#. Finding only 1-2 nodes with gsub, getting loads with gtop as expected.

   * duh, this is because of default of no recursion


::

    g4daenode.sh --debug 3155        


    In [78]: map(lambda _:(gsub.dae.xmlnode.getpath(_),_.attrib), gsub.dae.xmlnode.xpath("//c:node", namespaces=dict(c="http://www.collada.org/2005/11/COLLADASchema")) )
    Out[78]: 
    [('/*/*[5]/*',
      {'name': '__dd__Geometry__AD__lvOIL0xbf5e0b8', 'id': '__dd__Geometry__AD__lvOIL0xbf5e0b8'}),
     ('/*/*[6]/*/*[1]',
      {'name': '__dd__Geometry__AD__lvSST--pvOIL0xc241510', 'id': '__dd__Geometry__AD__lvSST--pvOIL0xc241510'})]

    In [87]: map(lambda _:(gsub.dae.xmlnode.getpath(_),_.attrib), gsub.dae.xmlnode.xpath("//*[local-name()='instance_node']", namespaces=dict(c="http://www.collada.org/2005/11/COLLADASchema")) )
    Out[87]: [('/*/*[6]/*/*[1]/*[2]', {'url': '#__dd__Geometry__AD__lvOIL0xbf5e0b8'})]


::

    In [5]: gtop.orig
    Out[5]: <Collada geometries=249>

    In [6]: gtop.orig.xmlnode
    Out[6]: <lxml.etree._ElementTree at 0x1030940e0>

    In [8]: gtop.orig.xmlnode.xpath("//*[local-name()='node']")
    Out[8]: 
    [<Element {http://www.collada.org/2005/11/COLLADASchema}node at 0x103593248>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}node at 0x103593200>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}node at 0x103593050>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}node at 0x103498ef0>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}node at 0x10349a2d8>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}node at 0x10349a518>,


    In [16]: d_ = lambda _:(gtop.orig.xmlnode.getpath(_),_.attrib)



::

    g4daenode.sh --debug 3155___3  

    In [22]: fnodes = filter(lambda _:_.find("{http://www.collada.org/2005/11/COLLADASchema}matrix") != None and _.find("{http://www.collada.org/2005/11/COLLADASchema}instance_node") != None, nodes) 

    In [53]: wnodes = filter(lambda _:_.xpath("./*[1]")[0].tag != '{http://www.collada.org/2005/11/COLLADASchema}matrix', fnodes)

    In [54]: len(wnodes)
    Out[54]: 11

    In [55]: len(nodes)
    Out[55]: 651

    In [56]: len(fnodes)
    Out[56]: 582


    In [68]: list(wnodes[10])
    Out[68]: 
    [<Element {http://www.collada.org/2005/11/COLLADASchema}instance_node at 0x10c656a28>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}matrix at 0x108c75dd0>]

    In [70]: list(fnodes[0])
    Out[70]: 
    [<Element {http://www.collada.org/2005/11/COLLADASchema}matrix at 0x108938830>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}instance_node at 0x10c66c050>]






Met this before
~~~~~~~~~~~~~~~~

* ~/env/geant4/geometry/collada/daediff.py 
* http://stackoverflow.com/questions/8385358/lxml-sorting-tag-order


::

    In [84]: wnodes[0]
    Out[84]: <Element {http://www.collada.org/2005/11/COLLADASchema}node at 0x10c683560>

    In [85]: list(wnodes[0])
    Out[85]: 
    [<Element {http://www.collada.org/2005/11/COLLADASchema}instance_node at 0x10c683440>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}matrix at 0x10895a680>]

    In [86]: w = wnodes[0]

    In [87]: w
    Out[87]: <Element {http://www.collada.org/2005/11/COLLADASchema}node at 0x10c683560>

    In [88]: w[:] = list(reversed(list(w)))

    In [89]: list(w)
    Out[89]: 
    [<Element {http://www.collada.org/2005/11/COLLADASchema}matrix at 0x10895a680>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}instance_node at 0x10c683440>]

    In [90]: list(wnodes[0])
    Out[90]: 
    [<Element {http://www.collada.org/2005/11/COLLADASchema}matrix at 0x10895a680>,
     <Element {http://www.collada.org/2005/11/COLLADASchema}instance_node at 0x10c683440>]



Default Color
~~~~~~~~~~~~~~

::

    gsub.dae.xmlnode.xpath("//c:effect/c:profile_COMMON/c:technique/c:phong/c:ambient/c:color", namespaces=NAMESPACES)



Schema Validation
------------------

* attribute 'id': '/dd/Materials/PPE0x92996b8' is not a valid value of the atomic type 'xs:ID'
* attribute 'sid': '/dd/Materials/PPE0x92996b8' is not a valid value of the atomic type 'xs:NCName'

Curiously get 'xs:ID' validation errors on G, but not N 

* http://www.schemacentral.com/sc/xsd/t-xsd_ID.html

  * The type xsd:ID is used for an attribute that uniquely identifies an element in an XML document. 


fails without network even with `--nonet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:dae blyth$ xmllint --nonet --noout --schema /Users/blyth/env/geant4/geometry/DAE/schema/collada_schema_1_4.xsd 3155___2.dae
    I/O error : Attempt to load network entity http://www.w3.org/2001/03/xml.xsd
    warning: failed to load external entity "http://www.w3.org/2001/03/xml.xsd"
    /Users/blyth/env/geant4/geometry/DAE/schema/collada_schema_1_4.xsd:22: element import: Schemas parser warning : Element '{http://www.w3.org/2001/XMLSchema}import': Failed to locate a schema at location 'http://www.w3.org/2001/03/xml.xsd'. Skipping the import.
    /Users/blyth/env/geant4/geometry/DAE/schema/collada_schema_1_4.xsd:201: element attribute: Schemas parser error : attribute use (unknown), attribute 'ref': The QName value '{http://www.w3.org/XML/1998/namespace}base' does not resolve to a(n) attribute declaration.
    WXS schema /Users/blyth/env/geant4/geometry/DAE/schema/collada_schema_1_4.xsd failed to compile
    delta:dae blyth$ 


::

    delta:dae blyth$ ll /System/Library/Frameworks/SceneKit.framework/Versions/A/Resources/
    total 208
    -rw-r--r--   1 root  wheel    6477 Sep 13  2013 SCNKitTypeInspector.nib
    -rw-r--r--   1 root  wheel    7544 Sep 13  2013 SCNKitSceneGraphView.nib
    -rw-r--r--   1 root  wheel    8065 Sep 13  2013 SCNKitRenderPassView.nib
    -rw-r--r--   1 root  wheel    5915 Sep 13  2013 SCNKitLibraryView.nib
    drwxr-xr-x   4 root  wheel     136 Oct  5  2013 BridgeSupport
    -rw-r--r--   1 root  wheel    4726 Oct  6  2013 xml.xsd
    -rw-r--r--   1 root  wheel   32054 Oct  6  2013 SCNRendererOptions.nib
    -rw-r--r--   1 root  wheel    5343 Oct  6  2013 SCNMonitor.nib
    -rw-r--r--   1 root  wheel   11100 Oct  6  2013 SCNKitImagePackerView.nib
    -rw-r--r--   1 root  wheel  413957 Oct  6  2013 COLLADASchema.xsd
    -rw-r--r--   1 root  wheel     457 Jul 14 16:04 version.plist
    -rw-r--r--   1 root  wheel    1079 Jul 14 16:04 Info.plist
    drwxr-xr-x   7 root  wheel     238 Jul 14 16:05 ..
    drwxr-xr-x  14 root  wheel     476 Jul 14 16:05 .






Strip Extras
-----------------

The extra nodes need to be stripped to avoid schema validation errors (why are extra content not ignored ?)::

    dae-strip-extra g4_00.dae.6 > g4_00.dae.6.noextra
    dae-validate  g4_00.dae.6.noextra


Big DAE crash Preview.app
--------------------------

::

    export-cd

    simon:juno blyth$ export-strip-extra-meta test3.dae
    orig test3.dae nometa test3.nometa.dae
    simon:juno blyth$ open test3.nometa.dae
    simon:juno blyth$ open test.nometa.dae   # this one is much smaller and opens OK

::

    Exception Type:  EXC_BAD_ACCESS (SIGSEGV)
    Exception Codes: KERN_INVALID_ADDRESS at 0x0000000000000038

    VM Regions Near 0x38:
    --> 
        __TEXT                 0000000100828000-0000000100a0c000 [ 1936K] r-x/rwx SM=COW  /Applications/Preview.app/Contents/MacOS/Preview

    Thread 0 Crashed:: Dispatch queue: com.apple.main-thread
    0   com.apple.SceneKit              0x00007fff8cb33dac C3DGenericSourceGetAccessor + 4
    1   com.apple.SceneKit              0x00007fff8ca81b39 __TransformAndAppendMeshSource + 31
    2   com.apple.SceneKit              0x00007fff8ca7f86c _C3DCreateFlattenedGeometryFromNodeHierarchy + 1253
    3   com.apple.SceneKit              0x00007fff8cb6289f C3DIOFinalizeLoadScene + 8367
    4   com.apple.SceneKit              0x00007fff8ca77eba C3DSceneSourceCreateSceneAtIndex + 642
    5   com.apple.SceneKit              0x00007fff8cb12821 -[SCNSceneSource _createSceneRefWithOptions:statusHandler:] + 331
    6   com.apple.SceneKit              0x00007fff8cb12df2 -[SCNSceneSource _sceneWithClass:options:statusHandler:] + 215
    7   com.apple.Preview               0x000000010087b082 0x100828000 + 340098
    8   com.apple.Preview               0x000000010082f754 0x100828000 + 30548
    9   com.apple.Preview               0x000000010082f3c7 0x100828000 + 29639
    10  libdispatch.dylib               0x00007fff8c1a91bb _dispatch_call_block_and_release + 12
    11  libdispatch.dylib               0x00007fff8c1a628d _dispatch_client_callout + 8
    12  libdispatch.dylib               0x00007fff8c1adef0 _dispatch_main_queue_callback_4CF + 333
    13  com.apple.CoreFoundation        0x00007fff88c8d4f9 __CFRUNLOOP_IS_SERVICING_THE_MAIN_DISPATCH_QUEUE__ + 9
    14  com.apple.CoreFoundation        0x00007fff88c48714 __CFRunLoopRun + 1636
    15  com.apple.CoreFoundation        0x00007fff88c47e75 CFRunLoopRunSpecific + 309
    16  com.apple.HIToolbox             0x00007fff8807ea0d RunCurrentEventLoopInMode + 226
    17  com.apple.HIToolbox             0x00007fff8807e7b7 ReceiveNextEventCommon + 479
    18  com.apple.HIToolbox             0x00007fff8807e5bc _BlockUntilNextEventMatchingListInModeWithFilter + 65
    19  com.apple.AppKit                0x00007fff8e4de24e _DPSNextEvent + 1434
    20  com.apple.AppKit                0x00007fff8e4dd89b -[NSApplication nextEventMatchingMask:untilDate:inMode:dequeue:] + 122
    21  com.apple.AppKit                0x00007fff8e4d199c -[NSApplication run] + 553
    22  com.apple.AppKit                0x00007fff8e4bc783 NSApplicationMain + 940
    23  libdyld.dylib                   0x00007fff8723a5fd start + 1




Code Organisation
--------------------

GDML inheritance heirarchy::

     G4GDMLWriteStructure
     G4GDMLWriteParamvol 
     G4GDMLWriteSetup
     G4GDMLWriteSolids
     G4GDMLWriteMaterials 
     G4GDMLWriteDefine        
     G4GDMLWrite              

DAE inheritance heirarchy::

     G4DAEWriteStructure   library_nodes         (based on G4GDMLWriteStructure)
     G4DAEWriteParamvol
     G4DAEWriteSetup       library_visual_scenes (based on G4GDMLWriteSetup)
     G4DAEWriteSolids      library_geometries    (based on G4GDMLWriteSolids) 
     G4DAEWriteMaterials   library_materials     (based on G4GDMLWriteMaterials) 
     G4DAEWriteEffects     library_effects       (originated) 
     G4DAEWriteAsset       asset                 (based on G4GDMLWriteDefine)             
     G4DAEWrite            COLLADA    

Maybe can restructure making G4DAE* chain of classes to inherit 
from the G4GDML* ones. This would avoid duplication and allow
G4DAE to incorporate GDML into extra tags.


FUNCTIONS
----------

dae-get
        grab latest .dae from N via http


LCG Builder-ization
---------------------

Ape the geant4 build with ::

    fenv
    cd $SITEROOT/lcgcmt/LCG_Builders/geant4/cmt
    cmt config
    . setup.sh

    cmt pkg_get      # looks to not use the script
    cmt pkg_config
    cmt pkg_make
    cmt pkg_install

Considered a separate LCG_Builder for g4dae, but that 
makes things more complicated. It is essentially a
rather extended patch against Geant4.

See also geant4/geometry/export/nuwa_integration



EOU
}
dae-dir(){ echo $(env-home)/geant4/geometry/DAE ; }
dae-cd(){  cd $(dae-dir); }
dae-mate(){ mate $(dae-dir) ; }


dae-strip-extra(){
  xsltproc $(env-home)/geant4/geometry/collada/strip-extra.xsl $1 
}

dae-lget(){
   local dae=${1}.dae
   local sdae=s${1}.dae
   local cmd="curl -sO http://localhost/g4dae/tree/${dae}"

   [ -d "$dae" ] && echo $msg dae $dae exists already && return
   [ ! -f "$dae" ] && echo $cmd && eval $cmd

   dae-strip-extra $dae > $sdae 
}


dae-install(){
   dae-install-lib
   dae-install-inc
}

dae-install-inc(){
   nuwa-
   local blib=$(env-home)/geant4/geometry/DAE/include
   local ilib=$(nuwa-g4-idir)/include
   ls -l $blib
   #ls -l $ilib
   local cmd="cp $blib/G4DAE*.{hh,icc} $ilib/"
   echo "$cmd"
   eval $cmd
   ls -l $ilib/G4DAE*
}

dae-install-lib(){
   nuwa-
   local name=libG4DAE.so
   local blib=$(nuwa-g4-bdir)/lib/Linux-g++/$name
   local ilib=$(nuwa-g4-idir)/lib/$name
   local cmd="cp $blib $ilib"
   echo $cmd
   eval $cmd
   ls -l $blib $ilib
}


dae-make(){
   local tgt=$1
   nuwa-
   make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_DAE=1 G4LIB_USE_DAE=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) G4INSTALL=$(nuwa-g4-bdir) CPPVERBOSE=1  $tgt
}


dae-switch(){
   perl -pi -e 's,GDML,DAE,g' *.*
}

dae-mv(){
   local name
   local newname
   local cmd
   ls -1 G4GDML*.* | while read name ; do
      newname=${name/GDML/DAE}
      cmd="mv $name $newname"
      echo $cmd
   done
}

dae-validate(){
   local pth=${1:-$(dae-pth)}
   local cmd="xmllint --noout --schema $(dae-xsd) $pth"
   echo $cmd
   eval $cmd
}

dae-lvalidate(){
   local pth=${1:-$(dae-pth)}
   local cmd="xmllint --noout --schema $(dae-lxsd) $pth"
   echo $cmd
   eval $cmd
}




dae-name(){ echo g4_01.dae ; }
dae-xsd(){  echo $(env-home)/geant4/geometry/DAE/schema/collada_schema_1_4.xsd  ; }
dae-url(){  echo http://belle7.nuu.edu.tw/dae/$(dae-name) ; }
dae-pth(){  echo $LOCAL_BASE/env/geant4/geometry/xdae/$(dae-name) ; }
dae-edit(){ vi $(dae-pth) ; }
dae-get(){
   local url=$(dae-url)
   local pth=$(dae-pth)
   local nam=$(basename $url)
   local cmd="curl -o $pth $url "
   echo $cmd
   eval $cmd

   dae-info
}

dae-info(){
   local pth=$(dae-pth)
   ls -l $pth
   du -h $pth
   wc -l $pth
}

