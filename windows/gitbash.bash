# === func-gen- : windows/gitbash fgp windows/gitbash.bash fgn gitbash fgh windows
gitbash-src(){      echo windows/gitbash.bash ; }
gitbash-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gitbash-src)} ; }
gitbash-vi(){       vi $(gitbash-source) ; }
gitbash-env(){      elocal- ; }
gitbash-usage(){ cat << EOU


* https://danlimerick.wordpress.com/2011/07/23/git-for-windows-tip-how-to-copy-and-paste-into-bash/

* http://www.hanselman.com/blog/Console2ABetterWindowsCommandPrompt.aspx






EOU
}
gitbash-dir(){ echo $(local-base)/env/windows/windows-gitbash ; }
gitbash-cd(){  cd $(gitbash-dir); }
gitbash-mate(){ mate $(gitbash-dir) ; }
gitbash-get(){
   local dir=$(dirname $(gitbash-dir)) &&  mkdir -p $dir && cd $dir

}
