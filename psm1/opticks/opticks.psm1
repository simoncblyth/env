
function opticks-src { "${env:userprofile}\env\psm1\opticks\opticks.psm1" }
function opticks-vi {    vim $(opticks-src) }
function opticks-prefix {  "${env:userprofile}\local\opticks" }
function opticks-check { Write-Host "check" }
function opticks-usage { Write-Host @"

Opticks Windows Powershell Definitions
========================================




"@ }



Export-ModuleMember -Function "opticks-*"



