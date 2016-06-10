
function opticks-src { return "${env:homepath}\env\windows\powershell\opticks.ps1" }
function opticks-prefix { return "${env:homepath}\local\opticks" }
function opticks-vi { vim $(opticks-src) }
function opticks-check { Write-Host "check" }
function opticks-usage { Write-Host @"

Update with::

  . $(opticks-src)


"@ }





