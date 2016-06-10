# === func-gen- : windows/powershell fgp windows/powershell.bash fgn powershell fgh windows
powershell-src(){      echo windows/powershell.bash ; }
powershell-source(){   echo ${BASH_SOURCE:-$(env-home)/$(powershell-src)} ; }
powershell-vi(){       vi $(powershell-source) ; }
powershell-env(){      elocal- ; }
powershell-usage(){ cat << EOU

Windows Powershell
=======================


There are choco pkgs powershell and powershell4 for 
windows management framework 5.0 and 4.0.
But maybe should try the v2 that comes with Windows7 SP1 first 

* https://chocolatey.org/packages/PowerShell
* https://chocolatey.org/packages/powershell4


Windows copy/paste
---------------------

* select then copy: ctrl-c
* paste : shift-insert


Allow runing scripts
----------------------


::

   Set-ExecutionPolicy Unrestricted -Scope CurrentUser

   get-host  # find version of powershell


HW
----

::


    PS C:\Users\ntuhep\env\windows\powershell> type hello.ps1
    Write-Host "Hello World"

    PS C:\Users\ntuhep\env\windows\powershell> & .\hello.ps1
    Hello World



Create Profile
-------------------

::

    New-Item -path $profile -type file -force

    vim $profile


Recursive rmdir
-------------------

::

    Remove-Item -Recurse -Force some-dir

    cmd /c "rd /s /q some-dir"








EOU
}
powershell-dir(){ echo $(local-base)/env/windows/windows-powershell ; }
powershell-cd(){  cd $(powershell-dir); }
powershell-mate(){ mate $(powershell-dir) ; }
powershell-get(){
   local dir=$(dirname $(powershell-dir)) &&  mkdir -p $dir && cd $dir

}
