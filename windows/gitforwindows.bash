# === func-gen- : windows/gitforwindows fgp windows/gitforwindows.bash fgn gitforwindows fgh windows
gitforwindows-src(){      echo windows/gitforwindows.bash ; }
gitforwindows-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gitforwindows-src)} ; }
gitforwindows-vi(){       vi $(gitforwindows-source) ; }
gitforwindows-env(){      elocal- ; }
gitforwindows-usage(){ cat << EOU


Git for Windows
=================

* https://gitforwindows.org
* https://github.com/git-for-windows/git/releases/latest

Hmm from Windows the download points to .exe that fails 
to download. Tried via github and get the same thing.






EOU
}
gitforwindows-dir(){ echo $(local-base)/env/windows/windows-gitforwindows ; }
gitforwindows-cd(){  cd $(gitforwindows-dir); }
gitforwindows-mate(){ mate $(gitforwindows-dir) ; }
gitforwindows-get(){
   local dir=$(dirname $(gitforwindows-dir)) &&  mkdir -p $dir && cd $dir

}
