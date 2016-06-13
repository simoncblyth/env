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
     [string]$config = "Debug"
  ) 

   # Debug Release MinSizeRel RelWithDebInfo 

   pushd $(g4-bdir)
   echo $pwd

   cmake --build . --config $config --target $target 


   popd
}




Export-ModuleMember -Function "g4-*"

