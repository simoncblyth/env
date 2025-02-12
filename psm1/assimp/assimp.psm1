function assimp-src { $script:MyInvocation.MyCommand.Path }
function assimp-sdir { Split-Path -Parent -Path $script:MyInvocation.MyCommand.Path }
function assimp-scd { cd $(assimp-sdir) }
function assimp-vi{   vim $(assimp-src) }
function assimp-usage { Write-Host @"

assimp powershell module
================================

::

   ipmo -fo -disablenamecheck assimp 
      # add line like above to profile using "vip" alias  

   ipmo -fo -disablenamecheck assimp ; assimp-usage
      # repeat above line while developing module function 

   src       $(assimp-src) 
   sdir      $(assimp-sdir) 


   tag       $(($info).tag)
   url       $(($info).url)

   filename  $(($info).filename)
   ext       $(($info).ext)
   name      $(($info).name)
   zip       $(($info).zip)
   dir       $(($info).dir)
   fold      $(($info).fold)


"@}


$tag = "assimp"
$url = "http://github.com/simoncblyth/assimp.git"
$info =  $(dist-info $tag $url) 


ipmo -fo -disablenamecheck dist

function assimp-info { $info }
function assimp-get  { dist-get $tag $url }

