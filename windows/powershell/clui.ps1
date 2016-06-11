function clui-usage{  echo @"


. `$(clui-src)
  # update

clui-vi

clui-open .\ReadMe.html
  # open in chrome   


"@
}

function clui-which
{
   param(
     [string]$cmd = "vim"
  )
    $c = get-command $cmd
    $c.Definition
}

function clui-src{ "${env:homepath}\env\windows\powershell\clui.ps1" }
function clui-vi{  vim $(clui-src) }

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






