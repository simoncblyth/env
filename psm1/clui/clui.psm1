function clui-src{ "${env:homepath}\env\psm1\clui\clui.psm1" }
function clui-usage{  echo @"


. `$(clui-src)
  # update

clui-vi

clui-open .\ReadMe.html
  # open in chrome   



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





function clui-which
{
   param(
     [string]$cmd = "vim"
  )
    $c = get-command $cmd
    $c.Definition
}

function clui-vi{  vim $(clui-src) }
function clui-vip{  vim $profile }


function clui-script{ $MyInvocation.MyCommand.Path }



function clui-path
{
   $p = Get-ChildItem -Path env:path
   $p.Value.split(";")
}

function clui-find
{
   param([string]$ptn = "*ssh*" )
   Get-ChildItem $pwd -Recurse -Filter $ptn
}

function clui-regs
{
   Get-ChildItem HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall
}

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

function clui-chrome
{
   "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
}
function clui-open
{
  param([string]$url = "stackoverflow.com")
  & $(clui-chrome) $url 
}


function clui-git-bash { "C:\Program Files\Git\git-bash.exe" }
function clui-git-bash-run{ & $(clui-git-bash) }



function clui-append-psmodulepath {
   param([string]$abspath)

   $p = ${env:psmodulepath}
   $p += ";$abspath"
   ${env:PSModulePath} = $p

   #$p = [Environment]::GetEnvironmentVariable("PSModulePath")
   #[Environment]::SetEnvironmentVariable("PSModulePath",$p)
}


New-Alias which clui-which
New-Alias vi $(clui-which vim)
New-Alias gitbash clui-git-bash-run
New-Alias vip clui-vip
New-Alias open clui-open



Export-ModuleMember -Function "clui-*"  -Alias *



