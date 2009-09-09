# === func-gen- : tw/twdev fgp tw/twdev.bash fgn twdev fgh tw
twdev-src(){      echo tw/twdev.bash ; }
twdev-source(){   echo ${BASH_SOURCE:-$(env-home)/$(twdev-src)} ; }
twdev-vi(){       vi $(twdev-source) ; }
twdev-env(){      elocal- ; }
twdev-usage(){
  cat << EOU
     twdev-src : $(twdev-src)
     twdev-dir : $(twdev-dir)

     twdev-rbase : $(twdev-rbase)
     twdev-repos : $(twdev-repos)


EOU
}
twdev-dir(){ echo $(local-base)/env/twdev ; }
twdev-cd(){  cd $(twdev-dir); }
twdev-mate(){ mate $(twdev-dir) ; }

twdev-rbase(){ echo http://toscawidgets.org/hg ; }
twdev-repos(){ echo tw.jquery ; }
twdev-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(twdev-dir) &&  mkdir -p $dir && cd $dir
   local repo ; for repo in $(twdev-repos) ; do
       [ ! -d "$repo" ] && hg clone $(twdev-rbase)/$repo || echo $msg repo $repo  is already cloned
   done

}


twdev-install(){
    
   local msg="=== $FUNCNAME :"
   rum-
   [ "$(which python)" != "$(rum-dir)/bin/python" ] && echo $msg ABORT this must be run whilst inside the rum virtualenv  && return 1

   local tips="tw.jquery"
   local tip ; for tip in $tips ; do
      twdev-cd
      cd $tip
      python setup.py develop
   done
}


