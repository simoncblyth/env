# === func-gen- : tools/hg2git/hgexporttool fgp tools/hg2git/hgexporttool.bash fgn hgexporttool fgh tools/hg2git src base/func.bash
hgexporttool-source(){   echo ${BASH_SOURCE} ; }
hgexporttool-edir(){ echo $(dirname $(hgexporttool-source)) ; }
hgexporttool-ecd(){  cd $(hgexporttool-edir); }
hgexporttool-dir(){  echo $LOCAL_BASE/env/tools/hg2git/hg-export-tool ; }
hgexporttool-cd(){   cd $(hgexporttool-dir); }
hgexporttool-vi(){   vi $(hgexporttool-source) ; }
hgexporttool-env(){  elocal- ; }
hgexporttool-usage(){ cat << EOU


* https://github.com/chrisjbillington/hg-export-tool

* see also fastexport-



EOU
}
hgexporttool-get(){
   local dir=$(dirname $(hgexporttool-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d "hg-export-tool" ] && git clone https://github.com/chrisjbillington/hg-export-tool.git 
}

hgexporttool-repomap-path(){ echo $HOME/repomap.json ; }
hgexporttool-authmap-path(){ echo $HOME/authors.map ; }

hgexporttool-repomap-vi(){ vi $(hgexporttool-repomap-path) ; }

hgexporttool-list-authors(){ 
   local msg="=== $FUNCNAME :"
   local authmap=$(hgexporttool-authmap-path)
   rm -f $authmap
   local repomap=$(hgexporttool-repomap-path)

   echo $msg repomap
   cat $repomap

   local cmd="python $(hgexporttool-dir)/list-authors.py $repomap"
   echo $msg cmd $cmd
   $cmd

   ls -l $authmap
   cat $authmap
}



hgexporttool-repolist-(){ cat << EOR
tracdev_hg
chroma_hg
g4dae_hg
g4dae-opticks_hg
heprez_hg
intro_to_cuda_hg
intro_to_numpy_hg
jnu_hg
mountains_hg
opticks-cmake-overhaul_hg
sphinxtest_hg
EOR
}


hgexporttool-repomap(){ 
   : generates a json repo map from the above repolist NB ordering is NOT preserved 
   local msg="=== $FUNCNAME :"
   local repomap=$(hgexporttool-repomap-path)
   echo $msg writing to repomap $repomap 
   hgexporttool-repolist- |  python $(hgexporttool-edir)/repomap.py > $repomap
   cat $repomap
}

hgexporttool-repolist-check(){
   : checks repo naming and directory existance matches expectations 
   local msg="=== $FUNCNAME :"
   local repos=$(hgexporttool-repolist-)
   local repo
   local name
   local rc=0
   for repo in $repos ; do 
       [ "${repo:(-3)}" != "_hg" ] && echo $msg ERROR repo $repo does not end with _hg && rc=1 
       name=${repo/_hg}
       [ -d "$name" ] && echo $msg ERROR directory name $name exists already && rc=1 
       #echo $msg repo $repo name $name
   done
   return $rc
}


