function vs-vi{   vim $env:userprofile\env\psm1\vs\vs.psm1 } 

function vs-usage{  echo @'
   vs-env
       copy Visual Studio 2015 environment into powershell


'@   

}


function vs-env
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

