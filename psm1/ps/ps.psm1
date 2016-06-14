function ps-src { $script:MyInvocation.MyCommand.Path }
function ps-sdir { Split-Path -Parent -Path $script:MyInvocation.MyCommand.Path }
function ps-scd { cd $(ps-sdir) }
function ps-vi{   vim $(ps-src) }
function ps-usage { Write-Host @"

ps powershell module
================================

::

   ipmo -fo -disablenamecheck ps 
      # add line like above to profile using "vip" alias  

   ipmo -fo -disablenamecheck ps ; ps-usage
      # repeat above line while developing module function 

   src       $(ps-src) 
   sdir      $(ps-sdir) 



"@}

#endtemplate above lines are used as template for powershell modules 


$base = "${env:userprofile}\env\psm1"


function ps-filltemplate
{
   param([string]$name="demo") 

   $lines = @()
   foreach($line in [io.file]::ReadAllLines($(ps-src)))
   {
      if ($line.Contains("endtemplate")) { break }
      $lines += $line -replace "ps",$name
   }
   return $lines
}

function ps-info
{
   param([string]$name="demo") 

   $dir =  [io.path]::combine($base,$name) 
   $path = [io.path]::combine($dir, $name + ".psm1" )

   @{ 
      name=$name;
      dir=$dir;
     path=$path;
    }
}

# ipmo -fo ps -disablenamecheck ; ps-usage

function ps-gen 
{ 
   param([string]$name="demo") 

   $info = ps-info $name
   $dir = $(($info).dir) 
   $path = $(($info).path) 


   if(![io.directory]::exists($dir) -and ![io.file]::exists($path))
   { 
       $lines = ps-filltemplate $name

       echo "Writing lines to path $path"

       md $dir > $null 

       $lines | out-file -encoding ascii $path
   }
   else
   {
       echo "Module already exists at $path"
   }
}



function ps-profile { echo @'


$oldhome = "C:\msys64\home\ntuhep"
$env:LOCAL_BASE = "\usr\local"

$env:PSModulePath += ";${env:userprofile}\env\psm1\"


Import-Module -Force -DisableNameChecking ps
function imp { param([string]$name) Import-Module -Force -DisableNameChecking $name }
function gen { param([string]$name) ps-gen $name ; imp $name }

imp vs
imp e
imp clui
imp dist
imp opticks
imp xercesc
imp g4


. $env:userprofile\env\tools\cmakewin.ps1

'@
}


