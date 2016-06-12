function e-src{ "${env:userprofile}\env\psm1\e\e.psm1" }
function e-usage{  echo @"


"@
}

function e-dir{ "${env:userprofile}\env" }
function e-cd{  cd $(e-dir) ; hg status }



New-Alias e e-cd 



Export-ModuleMember -Function "e-*"  -Alias *





