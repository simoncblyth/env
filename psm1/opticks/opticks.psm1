function opticks-src { "${env:userprofile}\env\psm1\opticks\opticks.psm1" }
function opticks-vi {    vim $(opticks-src) }
function opticks-prefix-former {  "${env:userprofile}\local\opticks" }
function opticks-prefix {  "${env:homepath}\local\opticks" }
function opticks-check { Write-Host "check" }
function opticks-usage { Write-Host @"

Opticks Windows Powershell Definitions
========================================

    ipmo -fo -disablenamecheck opticks ; opticks-usage 

    prefix : $(opticks-prefix)
    former : $(opticks-prefix-former)


"@ }



Export-ModuleMember -Function "opticks-*"



