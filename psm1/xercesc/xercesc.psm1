
Import-Module dist    -DisableNameChecking


$tag = "xercesc"
$url = "http://ftp.mirror.tw/pub/apache//xerces/c/3/sources/xerces-c-3.1.3.zip"
$info = $(dist-info $tag $url)

function xercesc-{  
   # this fails to update from here within a function but the command works from powershell
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
 

"@ }


function xercesc-dir{  $info.Item("dir") }
function xercesc-fold{ $info.Item("fold") }
function xercesc-cd{  cd $(xercesc-dir)  }
function xercesc-fcd{ cd $(xercesc-fold) }

function xercesc-get{ dist-get $tag $url }
function xercesc-sln{  "$(xercesc-dir)\projects\Win32\VC12\xerces-all\xerces-all.sln" }



Export-ModuleMember -Function "xercesc-*"

