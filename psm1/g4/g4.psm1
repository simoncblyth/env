function g4-src { $script:MyInvocation.MyCommand.Path }
function g4-sdir { Split-Path -Parent -Path $script:MyInvocation.MyCommand.Path }
function g4-vi{  vim $(g4-src) }
function g4-usage { Write-Host @"

To update these functions in callers scope use the
below ipmo command, putting that in function doesnt update parent scope::


   ipmo -fo g4 -disablenamecheck 
   ipmo -fo g4 -disablenamecheck ; g4-cmake 
        # repeat above while changing function of a module

   src       $(g4-src) 
   sdir      $(g4-sdir) 


   tag       $(($info).tag)
   url       $(($info).url)

   filename  $(($info).filename)
   name      $(($info).name)
   zip       $(($info).zip)
   dir       $(($info).dir)
   fold      $(($info).fold)



Hmm too may symbols in processes lib when using Debug::

    "C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\install.vcxproj" (default target) (1) ->
    "C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\ALL_BUILD.vcxproj" (default target) (3) ->
    "C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\source\visualization\FukuiRenderer\G4FR.vcxproj" (default target) (7) ->
    "C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\source\event\G4event.vcxproj" (default target) (27) ->
    "C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\source\processes\G4processes.vcxproj" (default target) (30) ->
    (Link target) ->
      LINK : fatal error LNK1189: library limit of 65535 objects exceeded [C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\source\processes\G4processes.vcxproj]

        31 Warning(s)
        1 Error(s)

    Time Elapsed 00:42:29.39


* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/installconfig/1778.html?inline=1


Switching to RelWithDebInfo get further::

    "C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\install.vcxproj" (default target) (1) ->
    "C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\ALL_BUILD.vcxproj" (default target) (3) ->
    "C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\source\persistency\G4persistency.vcxproj" (default target) (64) ->
    (Link target) ->
      \usr\local\env\windows\ome\xerces-c-3.1.3\Build\Win32\VC14\Debug\xerces-c_3_1D.dll : fatal error LNK1107: invalid or corrupt file: cannot read at 0x368 [C:\Users\ntuhep\local\opticks\externals\g4\geant4_10_02_p01.build\source\persistency
    \G4persistency.vcxproj]

        21 Warning(s)
        1 Error(s)

    Time Elapsed 00:43:03.07


From G4persistency.vcxproj note that its the only dll all others are lib::

    <AdditionalDependencies>
                 kernel32.lib;user32.lib;gdi32.lib;winspool.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;comdlg32.lib;advapi32.lib;
                 ..\..\BuildProducts\Debug\lib\_G4persistency-archive.lib;
                 ..\..\BuildProducts\Debug\lib\G4run.lib;
                 \usr\local\env\windows\ome\xerces-c-3.1.3\Build\Win32\VC14\Debug\xerces-c_3_1D.dll;
                 ..\..\BuildProducts\Debug\lib\G4event.lib;
                 ..\..\BuildProducts\Debug\lib\G4tracking.lib;
                 ..\..\BuildProducts\Debug\lib\G4processes.lib;
                 ..\..\BuildProducts\Debug\lib\G4digits_hits.lib;
                 ..\..\BuildProducts\Debug\lib\G4expat.lib;
                 ..\..\BuildProducts\Debug\lib\G4zlib.lib;
                 ..\..\BuildProducts\Debug\lib\G4track.lib;
                 ..\..\BuildProducts\Debug\lib\G4particles.lib;
                 ..\..\BuildProducts\Debug\lib\G4geometry.lib;
                 ..\..\BuildProducts\Debug\lib\G4graphics_reps.lib;
                 ..\..\BuildProducts\Debug\lib\G4materials.lib;
                 ..\..\BuildProducts\Debug\lib\G4intercoms.lib;
                 ..\..\BuildProducts\Debug\lib\G4global.lib;
                 ..\..\BuildProducts\Debug\lib\G4clhep.lib
    </AdditionalDependencies>

Checking xercesc products from xercesc-cd::

    PS C:\usr\local\env\windows\ome\xerces-c-3.1.3> gci -R -filter '*.lib'
    -a---         6/12/2016   5:27 PM    3128932 xerces-c_3D.lib


What are ilk pdb exp lib dll ?

    -a---         6/12/2016   5:29 PM      64000 StdInParse.exe
    -a---         6/12/2016   5:29 PM     594912 StdInParse.ilk
    -a---         6/12/2016   5:29 PM    1011712 StdInParse.pdb
    -a---         6/12/2016   5:29 PM      94208 ThreadTest.exe
    -a---         6/12/2016   5:29 PM    1001520 ThreadTest.ilk
    -a---         6/12/2016   5:29 PM    1175552 ThreadTest.pdb
    -a---         6/12/2016   5:27 PM    1937456 xerces-c_3D.exp
    -a---         6/12/2016   5:27 PM    3128932 xerces-c_3D.lib
    -a---         6/12/2016   5:27 PM    3998208 xerces-c_3_1D.dll
    -a---         6/12/2016   5:28 PM   29935912 xerces-c_3_1D.ilk
    -a---         6/12/2016   5:28 PM   10735616 xerces-c_3_1D.pdb
    -a---         6/12/2016   5:29 PM      43008 XInclude.exe
    -a---         6/12/2016   5:29 PM     361860 XInclude.ilk
    -a---         6/12/2016   5:29 PM     847872 XInclude.pdb
    -a---         6/12/2016   5:29 PM     124416 XSerializerTest.exe
    -a---         6/12/2016   5:29 PM     861224 XSerializerTest.ilk
    -a---         6/12/2016   5:29 PM    1265664 XSerializerTest.pdb




"@ }


Import-Module dist    -DisableNameChecking
Import-Module xercesc -DisableNameChecking


$tag = "g4"
$url = "http://geant4.cern.ch/support/source/geant4_10_02_p01.zip" 
$info = $(dist-info $tag $url)


function g4-info{  $info }
function g4-dir{  $info.Item("dir") }
function g4-bdir{ $info.Item("bdir") }
function g4-fold{ $info.Item("fold") }
function g4-prefix{ $info.Item("prefix") }

function g4-scd{ cd $(g4-sdir)  }
function g4-cd{  cd $(g4-dir)  }
function g4-fcd{ cd $(g4-fold) }
function g4-bcd{ cd $(g4-bdir) }


function g4-get
{
   dist-get $tag $url
}

function g4-wipe
{
   $bdir = g4-bdir
   rd -R -force $bdir | out-null    
}


# ipmo -fo g4 -disablenamecheck ; g4-configure 
function g4-cmake
{
    $bdir = g4-bdir
    md -f $bdir | out-null
    pushd $bdir 

    #echo $pwd
    cmake `
           "-DGEANT4_INSTALL_DATA=ON" `
           "-DGEANT4_USE_GDML=ON" `
           "-DXERCESC_LIBRARY=$(xercesc-lib)" `
           "-DXERCESC_INCLUDE_DIR=$(xercesc-include)" `
           "-DCMAKE_INSTALL_PREFIX=$(g4-prefix)" `
           $(g4-dir)


    popd
}

function g4-configure
{
    g4-wipe
    g4-cmake
}

function g4--
{
   param(
     [string]$target = "install",
     [string]$config = "RelWithDebInfo"
  ) 

   # Debug Release MinSizeRel RelWithDebInfo 

   pushd $(g4-bdir)
   echo " Building in $pwd config $config target $target " 

   cmake --build . --config $config --target $target 


   popd
}




Export-ModuleMember -Function "g4-*"

