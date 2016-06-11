
Import-Module opticks -DisableNameChecking


function g4-src { "${env:userprofile}\env\psm1\g4\g4.psm1" }
function g4-sdir{ [io.path]::GetDirectoryName($(g4-src))  }
function g4-vi{   vim $(g4-src) }
function g4-check { Write-Host "check" }
function g4-usage { Write-Host @"

To update these functions in callers powershell use the
below, seems doing that in above g4- function 
does not update in the scope of the caller.

Develop function with::

  . $(g4-src) ; g4-get
  g4-vi


"@ }


function g4-dir { "$(opticks-prefix)\externals\g4\$(g4-name)" }
function g4-fold{ [io.path]::GetDirectoryName($(g4-dir))  }

function g4-name{ "geant4_10_02_p01" }
function g4-url{  "http://geant4.cern.ch/support/source/$(g4-name).zip" }

function g4-scd{ cd $(g4-sdir)  }
function g4-cd{  cd $(g4-dir)  }
function g4-fcd{ cd $(g4-fold) }

function g4-nam{  ([System.Uri]$(g4-url)).Segments[-1] }
function g4-zip{  [io.path]::combine($(g4-fold), $(g4-nam)) }

function g4-get
{

   $fold = g4-fold
   mkdir -Force $fold > $null
   g4-fcd

   $url = g4-url
   $zip = g4-zip

   if(![io.file]::exists($zip))
   {
       Write-Host "Downloading $url to $zip "
       $wc = New-Object System.Net.WebClient
       $wc.DownloadFile($url,$zip)
   }
   else
   {
       Write-Host "Already downloaded $url to $zip "

   }

   $dir = g4-dir
   if(![io.directory]::exists($dir))
   {
       Write-Host "Extracting zip $zip into $dir "

       $shell = new-object -com shell.application
       $archive = $shell.namespace("C:$zip")
       $dest = $shell.namespace("C:$fold")

       #foreach ($item in $archive.items())
       #{
       #   Write-Host "Extracting $item"
       #   $dest.CopyHere($item)
       #}

       $dest.CopyHere($archive.items())

   }
   else
   {
       Write-Host "Already extracted zip $zip into $dir "
   }
}


Export-ModuleMember -Function "g4-*"

