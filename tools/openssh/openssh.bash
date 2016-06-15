# === func-gen- : tools/openssh/openssh fgp tools/openssh/openssh.bash fgn openssh fgh tools/openssh
openssh-src(){      echo tools/openssh/openssh.bash ; }
openssh-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openssh-src)} ; }
openssh-vi(){       vi $(openssh-source) ; }
openssh-env(){      elocal- ; }
openssh-usage(){ cat << EOU



OpenSSH
===========


OpenSSH depends on OpenSSL




* https://www.openssl.org/source/


Win32 port of OpenSSH
------------------------


* https://github.com/PowerShell/Win32-OpenSSH

* https://blogs.msdn.microsoft.com/powershell/2015/10/19/openssh-for-windows-update/

* https://github.com/PowerShell/Win32-OpenSSH/wiki

* https://github.com/PowerShell/Win32-OpenSSH/wiki/Win32-OpenSSH-Automated-Install-and-Upgrade-using-Chocolatey

* https://chocolatey.org/packages/win32-openssh/




EOU
}
openssh-dir(){ echo $(local-base)/env/tools/openssh/tools/openssh-openssh ; }
openssh-cd(){  cd $(openssh-dir); }
openssh-mate(){ mate $(openssh-dir) ; }
openssh-get(){
   local dir=$(dirname $(openssh-dir)) &&  mkdir -p $dir && cd $dir

}
