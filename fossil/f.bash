# === func-gen- : fossil/f fgp fossil/f.bash fgn f fgh fossil
f-src(){      echo fossil/f.bash ; }
f-source(){   echo ${BASH_SOURCE:-$(env-home)/$(f-src)} ; }
f-vi(){       vi $(f-source) ; }
f-env(){      
   elocal- ; 
   cfg- ;
   cfg-context fossil ~/.env.cnf  # generates and evaluates bash function based on ini file section
}
f-cfg(){ type _cfg_fossil  ; }   # dump config for debugging 
f-usage(){ cat << EOU

FOSSIL USAGE
=============

See `fossil-` for details on building, installation and serving.
This is to collect usage shortcuts and adhere to standard places
to keep fossil repos and wcdirs.

f-clone url name

   f-clone http://fossil.wanderinghorse.net/repos/cson/index.cgi cson

f-open name

   create working copy dir if it doesnt exist and `fossil open` the repo into it
   equivalent of `svn checkout` 
  
f-ui name

    local only access

f-web name 
  
    open the launctl configured daemon on server port : which is remotely accessible if firewall allows

    simon:cson blyth$ f-web cson
    open http://localhost:591/cson/

f-pw



EOU
}
f-dir(){ echo $repodir; }
f-scd(){ cd $(f-dir) ; }
f-cd(){
   local wcdir=$(f-wcdir $name) 
   cd $wcdir
}
f-ls(){ ls -l $repodir ; }

f-wcdir(){ echo $HOME/$1 ; }
f-repo(){ echo $repodir/$1.fossil ; }
f-lurl(){ echo http://localhost:$port/$1/ ; }
f-purl(){ echo http://localhost:$port/$1/setup_uedit?id=1 ; }
f-turd(){ echo .fslckout ;}
f-clone(){
   local rurl=$1
   local name=$2
   local repo=$(f-repo $name)
   local cmd="fossil clone $rurl $repo"
   [ -f "$repo" ] && echo $msg repo $repo is already present && f-ls && return 0
   echo $msg $cmd
}

f-ui(){  cd $(f-wcdir $1) ; fossil ui ; }
f-web(){ 
  local name=$1
  [ -z "$name" ] && echo $msg repo name is required && return 1
  local lurl=$(f-lurl $name)
  local cmd="open $lurl"
  echo $msg $cmd
  eval $cmd
}

f-pw(){ open $(f-purl $1) ; }

f-open(){
   local name=${1:-cson}
   local wcdir=$(f-wcdir $1)
   local repo=$(f-repo $1)

   [ ! -f "$repo" ] && echo $msg repo $repo does not exist && return 1

   if [ ! -d "$wcdir" ]; then
      echo $msg creating wcdir $wcdir
      mkdir -p $wcdir
   fi

   echo $msg cd to wcdir $wcdir
   cd $wcdir

   local turd=$(f-turd)
   if [ ! -f "$turd" ]; then 
       local cmd="fossil open $repo"
       echo $msg $cmd
       eval $cmd
   else
       echo $msg the turd $turd is already present 
   fi

}


