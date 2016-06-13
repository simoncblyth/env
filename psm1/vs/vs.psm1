function vs-src { $script:MyInvocation.MyCommand.Path }
function vs-vi{   vim $(vs-src) }
function vs-usage{  echo @'

Visual Studio 2015 Functions
=================================

vs-export
    export Visual Studio 2015 environment into powershell ENV

'@   

}


function vs-export
{
    $dir = $env:vs140comntools
    $bat = "vsvars32.bat"
    echo "Adopting $bat environment from $dir "
    pushd $dir
    cmd /c "$bat&set" | foreach {
        
        if($_ -match "=")
        {
           $v = $_.split("=")
           set-item -force -path "ENV:\$($v[0])" -value "$($v[1])" 

        }
    } 
    popd
}


Export-ModuleMember -Function "vs-*"

