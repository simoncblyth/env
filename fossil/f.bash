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

See :doc:`/fossil/fossil` for details on building, installation and serving.
This is to collect usage shortcuts and adhere to standard places
to keep fossil repos and wcdirs.


*f-clone url name*

   f-clone http://fossil.wanderinghorse.net/repos/cson/index.cgi cson
   f-clone http://www.fossil-scm.org  fossil 

*f-open name*

   create working copy dir if it doesnt exist, cd into it and `fossil open` the repo there.
   Equivalent of `svn checkout` 

   CAUTION `fossil open` unlike `svn checkout` acts in PWD,
   so this function creates a new `name` wcdir and  changes directory to it before opening
 
*f-ui name*

    local only access

*f-web name* 
  
    open the launctl configured daemon on server port : which is remotely accessible if firewall allows::

        simon:cson blyth$ f-web cson
        open http://localhost:591/cson/

*f-pw*
 
    open the password setting page in webinterface, in order to set it something more memorable
    than the randomlu assigned password that the `f-clone` yielded

*f-global*

    lists all repositories and checkouts on the node as recorded 
    in the global_config table of the ~/.fossil DB

*f-localtime name*

   http://www.mail-archive.com/fossil-users@lists.fossil-scm.org/msg08359.html
   Use localtime on timeline rather than UTC

   Hmm, didnt work::

       simon:env.f blyth$ fossil sql
       ...
       sqlite>  SELECT datetime('now','localtime') ;
       datetime('now','localtime')
       ---------------------------
       2013-04-02 11:55:33     


*f-sqlite3 name*

    command line sqlite3 shell connected to repo
    NB not precisely the same as running *fossil sql* from a checkout, eg note
    that localtime works::
     
        simon:e blyth$ f-sqlite3 env
        sqlite3 /var/scm/fossil/env.fossil
        ...
        sqlite> SELECT datetime('now','localtime') ;
        datetime('now','localtime')
        ---------------------------
        2013-04-02 19:59:45        
        sqlite> 


Fossil cloning SOP
-----------------------

#. run *f-clone url name* notice the generated password for the $USER
#. open web interface at http://localhost:591/name/
#. login as $USER using generated password 
#. click [Admin], [Users], [$USER]
#. edit the password to something memorable and click [Apply Changes], then [Login]

Fossil Config 
----------------

Kept in *~/.env.cnf* 

* server level in *[fossil]* section
* repo level in *[name.fossil]* sections



EOU
}
f-dir(){ echo $repodir; }
f-scd(){ cd $(f-dir) ; }
f-cd(){
   local wcdir=$(f-wcdir $name) 
   cd $wcdir
}
f-ls(){ ls -l $repodir ; }

f-global(){ 
   local init=~/.sqlite3_width_50_100
   [ ! -f "$init" ] && echo ".width 50 100" > $init
   echo "select * from global_config ;" | sqlite3 -init $init -column ~/.fossil
}
f-all(){
   echo REPO:
   fossil all ls
   echo CKOUT:
   fossil all ls --ckout
}

f-localtime(){
  local repo=$(f-repo ${1:-dummy})
  [ ! -f "$repo" ] && echo $msg no such repo at $repo && return 1 
  echo "REPLACE INTO config(name,value) VALUES('timeline-utc','2'); " | sqlite3 $repo
}


f-sqlite3(){
  local repo=$(f-repo ${1:-dummy})
  [ ! -f "$repo" ] && echo $msg no such repo at $repo && return 1 
  local cmd="sqlite3 $repo"
  echo $msg $cmd
  eval $cmd
}

f-wcdir(){ 
   local name=$1
   case $name in 
     env) echo $HOME/$name.f ;; 
       *) echo $HOME/$name  ;;
   esac 
}
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


