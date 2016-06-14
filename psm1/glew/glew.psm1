function glew-src { $script:MyInvocation.MyCommand.Path }
function glew-sdir { Split-Path -Parent -Path $script:MyInvocation.MyCommand.Path }
function glew-scd { cd $(glew-sdir) }
function glew-vi{   vim $(glew-src) }
function glew-usage { Write-Host @"

glew powershell module
================================

::

   ipmo -fo -disablenamecheck glew 
      # add line like above to profile using "vip" alias  

   ipmo -fo -disablenamecheck glew ; glew-usage
      # repeat above line while developing module function 

   src       $(glew-src) 
   sdir      $(glew-sdir) 


   tag       $(($info).tag)
   url       $(($info).url)
   override  $(($info).override)

   filename  $(($info).filename)
   name      $(($info).name)
   zip       $(($info).zip)
   dir       $(($info).dir)
   fold      $(($info).fold)


::

    PS C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0> gci .\*  -R -Include *.dll,*.lib,*.h | format-table Directory,Name

    Directory                                                                                                               Name
    ---------                                                                                                               ----
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\bin\Release\Win32                                              glew32.dll
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\bin\Release\x64                                                glew32.dll
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\bin\Release MX\Win32                                           glew32mx.dll
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\bin\Release MX\x64                                             glew32mx.dll
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\include\GL                                                     glew.h
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\include\GL                                                     glxew.h
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\include\GL                                                     wglew.h
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\lib\Release\Win32                                              glew32.lib
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\lib\Release\Win32                                              glew32s.lib
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\lib\Release\x64                                                glew32.lib
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\lib\Release\x64                                                glew32s.lib
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\lib\Release MX\Win32                                           glew32mx.lib
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\lib\Release MX\Win32                                           glew32mxs.lib
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\lib\Release MX\x64                                             glew32mx.lib
    C:\Users\ntuhep\local\opticks\externals\glew\glew-1.13.0\lib\Release MX\x64                                             glew32mxs.lib


"@}


ipmo -fo -disablenamecheck dist

$tag = "glew"
$url = "http://downloads.sourceforge.net/project/glew/glew/1.13.0/glew-1.13.0-win32.zip"
$override = "glew-1.13.0"
$info = $(dist-info $tag $url $override)



function glew-info { $info }
function glew-dir{  $info.Item("dir") }
function glew-bdir{ $info.Item("bdir") }
function glew-fold{ $info.Item("fold") }
function glew-prefix{ $info.Item("prefix") }

function glew-scd{ cd $(glew-sdir)  }
function glew-cd{  cd $(glew-dir)  }
function glew-fcd{ cd $(glew-fold) }
function glew-bcd{ cd $(glew-bdir) }


function glew-get { dist-get $tag $url }



function glew-wipe
{
   $bdir = glew-bdir
   rd -R -force $bdir | out-null
}




