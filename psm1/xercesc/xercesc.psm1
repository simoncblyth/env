Import-Module dist    -DisableNameChecking

$tag = "xercesc"
$url = "http://ftp.mirror.tw/pub/apache//xerces/c/3/sources/xerces-c-3.1.3.zip"
$info = $(dist-info $tag $url)

function xercesc-{  
   # this fails to update from here within a function but the command works from powershell
   # abbreviate with: ipmo -fo xercesc -DisableNameChecking
   Import-Module xercesc -DisableNameChecking -Force
}

function xercesc-src{ "${env:userprofile}\env\psm1\xercesc\xercesc.psm1" }
function xercesc-vi{   vim $(xercesc-src) }
function xercesc-info{  $info }
function xercesc-usage{ Write-Host @"

  tag : $tag
  url : $url
  info : $(xercesc-info)

  checking update again

To build:

      vs-env             # setup Visual Studio 2015 environment
      xercesc-build

With "/p:Configuration=ICU Debug" fails to find

    unicode/uloc.h
    unicode/uchar.h

Removing ICU get further, but "/p:Platform=x64" gives link error:

    \XercesLib\Base64.obj : fatal error LNK1112: module machine type 'X86' conflicts with target machine type 'x64'
 

Changing to "Win32" succeeds to build dll and many example .exe::

    PS C:\usr\local\env\windows\ome\xerces-c-3.1.3> gci -R -fi "*.dll"


        Directory: C:\usr\local\env\windows\ome\xerces-c-3.1.3\Build\Win32\VC14\Debug


    Mode                LastWriteTime     Length Name
    ----                -------------     ------ ----
    -a---         6/12/2016   2:27 AM    3998208 xerces-c_3_1D.dll




"@ }


function xercesc-get{ dist-get $tag $url }
function xercesc-cd{  cd $(xercesc-dir)  }
function xercesc-fcd{ cd $(xercesc-fold) }

#function xercesc-dir{  $info.Item("dir") }
#function xercesc-fold{ $info.Item("fold") }
#function xercesc-sln{  "$(xercesc-dir)\projects\Win32\VC12\xerces-all\xerces-all.sln" }

function xercesc-dir{   "\usr\local\env\windows\ome\xerces-c-3.1.3\" }
function xercesc-fold{  "\usr\local\env\windows\ome\" }
function xercesc-sln{   "$(xercesc-dir)\projects\Win32\VC14\xerces-all\xerces-all.sln" }


function xercesc-build{
    xercesc-cd
    msbuild $(xercesc-sln) "/p:Configuration=Debug" "/p:Platform=Win32" "/p:useenv=true" "/v:d"
}

function xercesc-lib{      "$(xercesc-dir)\Build\Win32\VC14\Debug\xerces-c_3_1D.dll" }
function xercesc-include{  "$(xercesc-dir)\src" } 



Export-ModuleMember -Function "xercesc-*"

