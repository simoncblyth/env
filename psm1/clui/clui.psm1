function global:clui- { 
   #Remove-Module clui  ;
   Import-Module clui  -Force  -DisableNameChecking
 }

function clui-src { $script:MyInvocation.MyCommand.Path }
function clui-dir { Split-Path -Parent -Path $script:MyInvocation.MyCommand.Path }
function clui-vi{  vim $(clui-src) }
function clui-vip{  vim $profile }
function clui-path { $p = Get-ChildItem -Path env:path ;  $p.Value.split(";") }

function clui-which { param( [string]$cmd = "vim") $c = get-command $cmd ; $c.Definition }
function clui-chrome { "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" }
function clui-git-bash { "C:\Program Files\Git\git-bash.exe" }
function clui-open { param([string]$url = "stackoverflow.com") & $(clui-chrome) $url }
function clui-git-bash-run{ & $(clui-git-bash) }

function clui-check{ "check9" }


New-Alias -f which clui-which
New-Alias -f vi $(clui-which vim)
New-Alias -f gitbash clui-git-bash-run
New-Alias -f vip clui-vip
New-Alias -f open clui-open
New-Alias -f ll Get-Child-Item


function clui-usage{  echo @"

Updating in parent scope::

   ipmo clui -fo -disablenamecheck ; clui-check



clui-vi

clui-open .\ReadMe.html
  # open in chrome   


http://stackoverflow.com/questions/5792746/add-powershell-function-to-the-parent-scope


https://powershellstation.com/2012/02/08/importing-modules-using-ascustomobject/


$c = ipmo clui -AsCustomObject -DisableNameChecking

 $c | gm    # dump the methods

 $c."clui-check"()    # huh not very convenient to invoke


  $m = get-module clui

  $m | gm  



"@
}

function clui-profile-example{ echo @'

$oldhome = "C:\msys64\home\ntuhep"

$env:PSModulePath += ";${env:userprofile}\env\psm1\"

Import-Module clui    -DisableNameChecking
Import-Module opticks -DisableNameChecking
Import-Module g4      -DisableNameChecking

'@
}

function clui-command-parameter-aliases
{
   param([string]$cmd = "ipmo")
   (get-command $cmd).parameters.values | select name, aliases
}
function clui-find
{
   param([string]$ptn = "*ssh*" )
   Get-ChildItem $pwd -Recurse -Filter "$ptn"
}

function clui-regs { Get-ChildItem HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall }
function clui-regs-dump
{
   param([string[]]$keys = @("DisplayName","InstallLocation") )
   $regs = clui-regs
   foreach($e in $regs){ 
       foreach($key in $keys)
       {
          echo $e.GetValue($key)
       }
  }
}



Export-ModuleMember -Function "clui-*"  -Alias *



