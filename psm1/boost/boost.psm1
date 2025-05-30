function boost-src { $script:MyInvocation.MyCommand.Path }
function boost-sdir { Split-Path -Parent -Path $script:MyInvocation.MyCommand.Path }
function boost-scd { cd $(boost-sdir) }
function boost-vi{   vim $(boost-src) }
function boost-usage { Write-Host @"

boost powershell module
================================

::

   imp boost 
      # add line like above to profile using "vip" alias  

   imp boost ; boost-usage
      # repeat above line while developing module function 

   src       $(boost-src) 
   sdir      $(boost-sdir) 


   tag       $(($info).tag)
   url       $(($info).url)

   filename  $(($info).filename)
   name      $(($info).name)
   zip       $(($info).zip)
   dir       $(($info).dir)
   fold      $(($info).fold)


"@}


Import-Module dist    -DisableNameChecking


$tag = "boost"
#$url = "http://downloads.sourceforge.net/project/boost/boost/1.61.0/boost_1_61_0.7z" 
$url = "http://downloads.sourceforge.net/project/boost/boost/1.61.0/boost_1_61_0.zip" 
$info = $(dist-info $tag $url)

function boost-info { $info }
function boost-dir{  $info.Item("dir") }
function boost-bdir{ $info.Item("bdir") }
function boost-fold{ $info.Item("fold") }
function boost-prefix{ $info.Item("prefix") }

function boost-scd{ cd $(boost-sdir)  }
function boost-cd{  cd $(boost-dir)  }
function boost-fcd{ cd $(boost-fold) }
function boost-bcd{ cd $(boost-bdir) }


function boost-get { dist-get $tag $url }

function boost-wipe
{
   $bdir = boost-bdir
   rd -R -force $bdir | out-null
}



