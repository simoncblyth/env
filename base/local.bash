
# suffering nameclash with heprez
elocal-src(){    echo base/local.bash ; }
elocal-source(){ echo ${BASH_SOURCE:-$ENV_HOME/$(elocal-src)} ; }
elocal-vi(){     vi $(elocal-source) ; }

local-src(){    echo base/local.bash ; }
local-source(){ echo ${BASH_SOURCE:-$ENV_HOME/$(local-src)} ; }
local-vi(){     vi $(local-source) ; }
local-systail(){ sudo tail -f  /var/log/system.log ; }
local-usage(){

cat << EOU

  These are used by most functions... and rarely need to be invoked directly by the user

   local-node        :  $(local-node)
   local-nodetag     :  $(local-nodetag)
   local-tag2node    :  $(local-tag2node)
   local-backup-tag  :  $(local-backup-tag)      paired backup node
   local-restore-tags:  $(local-restore-tags)    nodes that could be restored from  
   local-restore-tag :  $(local-restore-tag)     designated node (the first in the backup list) to restore from  
   local-mbackup-tag :  $(local-mbackup-tag)      locally mounted backup ... usually eg for C get .. BKP_C
   local-sudo        :  $(local-sudo)            is set on nodes which use system tools mostly
   
   local-system-base :  $(local-system-base)      
   local-base        :  $(local-base)
   local-prefix      :  $(local-prefix)
   local-var-base    :  $(local-var-base)
   local-scm-fold    :  $(local-scm-fold)
   local-user-base   :  $(local-user-base)
   local-output-base :  $(local-output-base)
   
   local-initialize 
       create the base folders + invoke trac-inherit-setup
   
   
   local-scm       : define the SCM_* coordinates of source code management node supporting the current node
    
   
     NODE_TAG_OVERRIDE : $NODE_TAG_OVERRIDE
     
     NODE_TAG      : $NODE_TAG
     LOCAL_BASE    : $LOCAL_BASE
     SYSTEM_BASE   : $SYSTEM_BASE     system tools like svn
     VAR_BASE      : $VAR_BASE 
     USER_BASE     : $USER_BASE
     OUTPUT_BASE   : $OUTPUT_BASE
                                
EOU

}

local-ip(){ curl ifconfig.me ; } # http://ifconfig.me/  determine external ip

local-info(){

   local t=${1:-$NODE_TAG}
   cat << EOI

   For tag $t   (actual node is $NODE_TAG) 

   local-server-tag  : $(local-server-tag $t)   node designated as the source node holding the repository
   local-restore-tag : $(local-restore-tag $t)  node holding the backup tarballs of the designated server node 
   local-backup-tag  : $(local-backup-tag $t)   paired node to which backups are sent from $t  

   local-system-base :  $(local-system-base $t)
   local-base        :  $(local-base $t)
   local-var-base    :  $(local-var-base $t)
   local-base-env    :  $(local-base-env $t)
   local-scm-fold    :  $(local-scm-fold $t)
   local-user-base   :  $(local-user-base $t)
   local-output-base :  $(local-output-base $t)

EOI

}

local-logpath(){
   case ${1:-$NODE_TAG} in
      *) echo /var/log/messages ;;
   esac
}
local-log(){    sudo vi $(local-logpath) ; }
local-tail(){   sudo tail -f $(local-logpath)  ; }
local-atail(){  sudo tail -f  /var/log/audit/audit.log ; }
local-alog(){   sudo vi /var/log/audit/audit.log ; }

local-static(){ 
   echo $HOME/simoncblyth.bitbucket.org
}

local-env(){

   local dbg=${1:-0}
   local msg="=== $FUNCNAME :"
   
   [ "$dbg" == "1" ] && echo $msg	 
          
   export SOURCE_NODE="g4pb"
   export SOURCE_TAG="G"

   export LOCAL_ARCH=$(uname)
   export LOCAL_NODE=$(local-node)
   export NODE_TAG=$(local-nodetag)       # glean where we are and define NODE_TAG
   export BACKUP_TAG=$(local-backup-tag)  # paired backup for the NODE_TAG 
   export SUDO=$(local-sudo)

  # local-scm        # assign coordinates of the SCM server for this node
 
   export SYSTEM_BASE=$(local-system-base) ## prequisite base for most everything, ie where to pick up subversion +
   export LOCAL_BASE=$(local-base)
   export STATIC_BASE=$(local-static)
   export ENV_PREFIX=$(local-prefix)

   export VAR_BASE=$(local-var-base)    ## operational files, like backups
   export SCM_FOLD=$(local-scm-fold)
   export VAR_BASE_BACKUP=$(local-var-base $BACKUP_TAG)
   export USER_BASE=$(local-user-base)
   export OUTPUT_BASE=$(local-output-base)



   # [ ! -d "$USER_BASE" ]   && echo "WARNING creating folder USER_BASE $USER_BASE" &&   mkdir -p $USER_BASE 
   # [ ! -d "$OUTPUT_BASE" ] && echo "WARNING creating folder OUTPUT_BASE $OUTPUT_BASE" &&   mkdir -p $OUTPUT_BASE 

    local-userprefs

}

local-tmp(){  echo /tmp ; }


local-node-deprecated(){
   case $NODE in 
     "") echo $(uname -a | cut -d " " -f 2 | cut -d "." -f 1) ;;
      *) echo $NODE ;;
    esac
}

local-host(){
   local h=$(hostname)
   echo ${h/.*/}
}

local-node(){
   case $NODE in 
     "" ) local-host ;;
       *) echo $NODE ;;
   esac    
}

local-userprefs(){
   case $USER in 
     blyth) export SVN_EDITOR=vi ;;
   esac
}


local-taginfo(){
   local tag=$1
   cat << EOI

    local-tag2node : $(local-tag2node $tag)
    local-tag2user : $(local-tag2user $tag)
    local-tag2ip   : $(local-tag2ip $tag)


EOI


}

local-port(){
   ## hmm maybe keep these in private-vals only ?
   private-
   case $1 in 
        apache) echo 80 ;;
slave-lighttpd) private-val SLAVE_LIGHTTPD_PORT  ;;  # info only ... 
 rabbitmq|amqp) private-val AMQP_PORT  ;;  # info only ... 
         mysql) private-val MYSQL_PORT ;;  # info only ... 
            hg) private-val HGWEB_PORT ;;
    dbi|plvdbi) private-val PLVDBI_PORT ;;
       runinfo) private-val RUNINFO_PORT ;;  
      ejabberd) private-val EJABBERD_PORT ;;  
 ejabberd-http) private-val EJABBERD_HTTP_PORT ;;  
             *) private-val $(echo ${1}_PORT | tr "[a-z]" "[A-Z"]) ;;
   esac  
}

local-tag2node(){
  case ${1:-$NODE_TAG} in 
          H) echo hfag  ;;
          M) echo mars  ;;
        C|S) echo cms01 ;;
 C2|SC2|C2R) echo cms02 ;;
         ZZ) echo dayabay-old ;;
         Y2) echo dayabay ;;
         YY) echo dyb1 ;;
         XX) echo dyb2 ;;
         H1) echo hep1 ;;
          N) echo belle7 ;;
         N1) echo belle1 ;;
    S2|P|G1) echo grid1 ;;
         G3) echo g3pb ;;
G|g4pb.local) echo g4pb ;; 
          *) echo unknown ;; 
  esac
}

local-tag2user(){
  case ${1:-$NODE_TAG} in
    SC2|S2|S) echo dayabayscp ;;
 H1|C|H|N|N1|C2) echo blyth ;;
          AB) echo aberdeen ;;
          AR) echo root ;;
         C2R) echo root ;;
           X) echo exist ;;
           P) echo dayabaysoft ;;
           L) echo sblyth ;;
    YY|XX|ZZ) echo root ;;
           *) echo $USER ;;
  esac 
}


local-scponly-tags(){   echo S SC2 S2 ; }
local-tags(){
   case ${1:-$NODE_TAG} in 
           G) echo M AB AR N N1 NE I BNL C C2 C2R H H1 X P G1 T LX  $(local-scponly-tags) ;;
           *) local-backup-tag ;;
   esac
}


local-ping(){ 
   local tag=$1
   shift 
   local cmd="ping $(local-tag2ip $tag) $*"
   echo $cmd
   eval $cmd
}

local-tag2ip(){
  case ${1:-$NODE_TAG} in
       AB|AR) echo aberdeentunnel.dyndns.org ;;
        SELF) echo 127.0.0.1 ;; 
         C|S) echo 140.112.101.190 ;; 
  C2|SC2|C2R) echo 140.112.101.191 ;;
     S2|G1|P) echo 140.112.102.250 ;;
          #N) echo belle7.nuu.edu.tw ;;
           N) echo 203.64.184.126 ;;
          N1) echo 203.64.184.127 ;;
          OM) echo 140.112.101.50 ;;
          NE) echo pdsf.nersc.gov ;;
       #    I) echo lxslc05.ihep.ac.cn ;;  lxslc05 needs klog and has flaky xauth 
           I) echo lxslc21.ihep.ac.cn ;;
          II) echo 140.112.101.199 ;;
         BNL) echo gateway.phy.bnl.gov ;;
         H|X) echo 140.112.101.48 ;;
          H1) echo 140.112.101.41 ;;
           G) echo 140.112.102.77 ;;
          BP) echo bpost.kek.jp ;; 
          LX) echo lxplus.cern.ch ;; 
           T) echo tersk.slac.stanford.edu ;;
           L) echo pal.nuu.edu.tw ;;
          YY) echo dyb1.ihep.ac.cn ;;
          XX) echo dyb2.ihep.ac.cn ;;
          ZZ) echo dayabay-old.ihep.ac.cn ;;
          Y2) echo dayabay.ihep.ac.cn ;;
         HKU) echo dayabay.physics.hku.hk ;;
         CUT) echo theta13.phy.cuhk.edu.hk ;;
         CUD) echo dayabay.phy.cuhk.edu.hk ;;
          WW) echo 202.122.32.203 ;;
         SDU) echo rh02.hepg.sdu.edu.cn ;; 
          CC) echo deliberately.trying.to.avoid.putting.in.repo ;; 
           K) echo 130.87.106.59 ;;
           M) echo dummy ;;
  esac 
}

local-port-sshd(){
   case ${1:-$NODE_TAG} in
       C) echo 22 ;;
      Z9) echo 229 ;;
     SDU) echo 22 ;;
       *) echo 22 ;;
    esac     
}

local-nodetag(){
  [ -n "$NODE_TAG_OVERRIDE" ] && echo $NODE_TAG_OVERRIDE && return 0
  case ${1:-$LOCAL_NODE} in
         g4pb) echo G ;;
  delta|simon) echo D ;;
         mars) echo M ;;
         coop) echo CO ;;
         hep1) echo H1 ;;
        hep52) echo H52 ;;
        cms01) echo C ;;
        cms02) local-nodetag-cms02 $USER ;;
      #dayabay) echo ZZ ;;
  dayabay-old) echo ZZ ;;
     dayabay1) echo Y1 ;;
      dayabay) echo Y2 ;;
  dayabay-new) echo Y2 ;;
        delta) echo D ;;
       #dybdb1) echo D1 ;;
       dybdb1) echo Y1 ;;
       dybdb2) echo D2 ;;
         dyb1) echo YY ;;
         dyb2) echo ZZ ;;
       belle7) echo N ;;
       belle1) echo N1 ;;
      gateway) echo B ;;
         g3pb) echo G ;;
          pal) echo L ;;
         hfag) local-nodetag-hfag $USER ;;
  thho-laptop) echo T ;;
 thho-desktop) echo T ;;
        hkvme) echo HKVME ;;
        grid1) local-nodetag-grid1 $USER ;;
          Arc) echo A ;;
         rh02) echo SDU ;;
          n98) echo CC ;;
       hgpu01) echo G1  ;;
     lxslc506) echo L6  ;;
     lxslc5??) echo LX  ;;
     ntuhep-PC) echo M ;;
            *) local-nodetag-other $(uname -n) ;;
  esac

}

local-nodetag-hfag(){
   case ${1:-$USER} in
      blyth) echo H ;;
       root) echo H ;;
      exist) echo X ;;
          *) echo U ;;
   esac
}

local-nodetag-cms02(){
   case ${1:-$USER} in
      blyth) echo C2 ;;
       root) echo C2R ;;
          *) echo U ;;
   esac
}

local-nodetag-grid1(){
   case ${1:-$USER} in 
     dayabaysoft|root) echo P ;;
                    *) echo G1 ;;
   esac
}

local-nodetag-other(){
   local host=${1:-$(uname -n)}
   if  [ "${host:0:6}" == "albert" ]; then   
        echo G1
   elif [ "${host:0:2}" == "pc" ]; then   
       echo N 
   elif [ "$host" == "dayabay.ihep.ac.cn" ]; then
       local-nodetag-xinchun    
   elif [ "$host" == "ihep" ]; then
       echo LT
   elif [ "$host" == "linux-h5h2" ]; then
       # DELL Precision 7510, openSUSE 42.1
       echo H5H2
   elif [ "${host:0:7}" == "junogpu" ]; then
       # JUNO GPU 00X
       echo GTL
   elif [ "$host" == "gputest.ihep.ac.cn" ]; then
       case $USER in
         lint) echo GTL ;;
         blyth) echo GTB ;;
       esac
   elif [ "${host:0:6}" == "lxslc6" ]; then
       case $USER in
         lint) echo GTL ;;
         blyth) echo GTB ;;
       esac
   else
       echo U
   fi
}

local-nodetag-xinchun(){
   case $USER in
     blyth) echo XT ;;
         *) echo XX ;;
   esac
}

local-sudo(){
  case ${1:-$NODE_TAG} in
  AA|HKU|G|H|T|C2|C|YY|N|ZZ|D) echo sudo ;;
      *) echo -n ;
  esac
}


local-backup-tag(){
   case ${1:-$NODE_TAG} in 
      G) echo A ;;
      H) echo C  ;;
      C) echo H1 C2 P H ;;
 C2|C2R) echo  C N H1 N1 ;;    ## exclude P as cannot get to work see #262 exclude H as out of space
 #C2|C2R) echo C ;;
      P) echo H1 C H N C2 ;;
     XX) echo ZZ SDU ;;
     ZZ) echo SDU ;;
     WW) echo SDU ;;
     AA) echo D8R ;;
    HKU) echo D8R CUT CUD ;;
     Y1) echo LT ;;
     Y2) echo Y1 ;;
     *) echo U ;;
   esac  
}

local-server-tag(){
   case ${1:-$NODE_TAG} in
     XX) echo ZZ ;;
     Y1) echo Y2 ;;
      *) echo P ;;  
   esac
}

local-restore-tags(){
   local tag=${1:-$NODE_TAG}
   echo $(local-backup-tag $(local-server-tag $tag))
}
local-restore-tag(){ local-first $(local-restore-tags $*) ; }
local-first(){        echo $1 ; }

local-email(){
   case ${1:-$NODE_TAG} in
     XX|YY|ZZ) echo maqm@ihep.ac.cn ;;
           Y1) echo lintao@ihep.ac.cn ;;
           Y2) echo lintao@ihep.ac.cn ;;
          HKU) echo jimngai@hku.hk ;;
            *) echo blyth@hep1.phys.ntu.edu.tw ;;
   esac
}

local-sshkeyholder(){
  case ${1:-$NODE_TAG} in
  C|C2|G|G1|N) echo blyth ;;
            P) echo dayabaysoft ;;
           XX) echo root ;;
           YY) echo maqm ;; 
           ZZ) echo maqm ;;
           *) echo blyth ;;
  esac

}


local-mbackup-disk(){
   case ${1:-$NODE_TAG} in 
     C|MBACKUP_C) echo /mnt/disk1 ;;
               *) echo NO_MBACKUP_DISK_DEFINED ;;
   esac
}

local-root(){
   case ${1:-$NODE_TAG} in
      C) echo -n ;;
      N) echo /data1 ;;
     N1) echo /data1 ;;
      M) echo /data1 ;;
     H1) echo /home/hep/blyth ;;
     *) echo -n ;;
   esac
}


local-base-workflow(){ echo $(local-base $*)/workflow ; }
local-base-env(){      echo $(local-base $*)/env ; }
local-base(){
    local t=${1:-$NODE_TAG}
    case $t in 
       WW) echo /home/blyth/local ;;
        G) echo /usr/local ;;
    OLD_G1) echo /disk/d3/dayabay/local ;;    ## used to be :  /data/w  then /disk/d4
       G1) echo /dyb/dybd07/user/blyth/hgpu01.ihep.ac.cn ;;  
    L6|LX) echo /dyb/dybd07/user/blyth/hgpu01.ihep.ac.cn ;;  
        P) echo /disk/d3/dayabay/local ;;
        L) echo /usr/local ;;
        H) echo /data/usr/local ;;
       Y1) echo /usr/local ;;
       Y2) echo /usr/local ;;
       H1) echo $(local-root $t)/local ;;
      H52) echo /data/local ;;
        T) echo /usr/local ;;
        N) echo $(local-root $t)/env/local ;;
       N1) echo $(local-root $t)/env/local ;;
    OLD_C) echo                         /data/env/local ;;
MBACKUP_C) echo $(local-mbackup-disk $t)/data/env/local ;;
        C) echo         $(local-root $t)/data/env/local ;;
   C2|C2R) echo         $(local-root $t)/data/env/local ;;
       XT) echo /home/tianxc ;;   
        K) echo /Users/heprez/local ;;
       G5) echo /home/blyth/local ;;
      GTL) echo ${MYENVTOP:-/afs/ihep.ac.cn/soft/juno/JUNO-ALL-SLC6/GPU/20150723}/local ;;
     H5H2) echo ${MYENVTOP:-/usr}/local ;;
      MGB) echo /c/usr/local ;;
        X) echo $HOME/local ;;
        *) echo /usr/local ;;
   esac
}


local-prefix(){ echo $(local-base)/env ; }
local-cd(){     cd $(local-prefix) ; }

local-system-base(){

   local t=${1:-$NODE_TAG}
   case $t in 
      P|G1) echo /disk/d4/dayabay/local ;;
     OLD_C) echo                         /data/env/system ;;
 MBACKUP_C) echo $(local-mbackup-disk $t)/data/env/system ;;
         C) echo $(local-root $t)/data/env/system ;;
    C2|C2R) echo $(local-root $t)/data/env/system ;;
         N) echo $(local-root $t)/env/system ;;
        H1) echo $(local-root $t)/system ;;
        XT) echo /home/tianxc/system ;;
        XX) echo /usr/local ;;
        YY) echo /usr/local ;;
        Y1) echo /usr/local ;;
        Y2) echo /usr/local ;;
        ZZ) echo /usr/local ;;
        D1) echo /soft  ;;
        D2) echo /soft  ;;
        CC) echo /home/guest/guest06/local ;;
         *) echo $(local-base $*) ;;
   esac
}





local-var-base(){
   local t=${1:-$NODE_TAG}
   case $t in 
        U) echo /var ;;
        P) echo /disk/d3/var ;;
       G1) echo /disk/d3/var ;;
        N) echo /var ;;
       XT) echo /home/tianxc ;; 
       XX) echo /home/zhuanghl ;; 
       YY) echo /home ;;
       Y1) echo /var ;;
       Y2) echo /var ;;
       ZZ) echo /home ;;
     IHEP) echo /home ;;  
    OLD_C) echo /var ;;
MBACKUP_C) echo $(local-mbackup-disk $t)/var ;;
        C) echo /data/var ;;
        S) echo /data/var ;;
        H) echo /data/var ;;
   A|B|Z9) echo /volume1/var ;;
       C2) echo $(local-root $t)/var ;;
       H1) echo $(local-root $t)/var ;;
  SDU_old) echo  /raid/dybsdu/dybbackup/var ;;
      SDU) echo  /raid4/dybsdu/dybbackup/var ;;
       LT) echo /tmp ;;
        *) echo  /var ;; 
   esac
}


local-scm-fold(){
   case ${1:-$NODE_TAG} in 
     WW) echo /home/scm ;;  
     Y1) echo /home/scm ;;  
     Y2) echo /home/scm ;;  
      *) echo $(local-var-base $*)/scm ;;
   esac
}

local-user-base(){
   case ${1:-$NODE_TAG} in
      G) echo $HOME/Work ;;
   P|G1) echo /disk/d3/$USER ;;
      L) echo $(local-base L) ;;
      H) echo $(local-base H) ;;
      T) echo $HOME/dybwork ;;
      N) echo $HOME ;;
     XT) echo /home/tianxc ;;  
      *) echo /tmp ;;
   esac
}

local-output-base(){
   case ${1:-$NODE_TAG} in
      N) echo /project/projectdirs/dayabay/scratch/$USER ;;
      *) echo $(local-user-base $*) ;;
   esac
}


local-mbackup(){
   local msg="=== $FUNCNAME :"
   local t=${1:-$NODE_TAG} 
   local locations="local-base local-system-base local-scm-fold"
   for loc in $locations ; do
      local-mbackup-     "$(eval $loc $t)" "$(eval $loc MBACKUP_$t)" 
      local-mbackup-chk- "$(eval $loc $t)" "$(eval $loc MBACKUP_$t)" 
   done
}


local-mbackup-(){
   local msg="=== $FUNCNAME :"
   local src=$1
   local dst=$2
   [ "$src" == "" ] && echo $msg ABORT src not defined for tag $t && return 1
   [ "$dst" == "" ] && echo $msg ABORT dst not defined for tag $t && return 1
   [ ! -d "$src" ]  && echo $msg ABORT src directory $src does not exist $t && return 1 
   [ ! -d "$dst" ]  && echo $msg WARNING creating destination directory $dst && $SUDO mkdir -p "$dst" && $SUDO chown $USER "$dst"

   local sudo
   [ "$src" == "/var/scm" ] && sudo=sudo

   local cmd="$sudo rsync --delete-after -razvt $src/ $dst/ " 
   echo $msg $(date) starting backup from src $src to dst $dst with cmd : $cmd 
   eval $cmd
   echo $msg $(date) completed

}

local-mbackup--(){
  screen bash -lc "local-mbackup " 
} 


local-mbackup-chk-(){
    local msg="=== $FUNCNAME :"
    local src=$1
    local dst=$2
    du -hs "$src"
    du -hs "$dst"
}



	
    
    
local-scm(){


########## SCM_* specify the source code repository coordinates #####################

 #export SCM_TAG="H"       ##      blyth@hfag      trac "production"  
 #export SCM_TAG="G"       ##      blyth@g4pb      trac testing

 # if SCM_TAG is set already use that value, otherwise default to H
 
 private-
 
SCM_TAG=${SCM_TAG:-H}
export SCM_TAG 

if [ "$SCM_TAG" == "P" ]; then
	
   SCM_HOST=grid1.phys.ntu.edu.tw
   SCM_PORT=6060
   SCM_USER=$USER
   SCM_PASS=$(private-val NON_SECURE_PASS)
   SCM_TRAC=env
   SCM_GROUP=GRID1
   
   SCM_URL=http://$SCM_HOST:$SCM_PORT
   
elif [ "$SCM_TAG" == "H" ]; then 

   #SCM_HOST=hfag.phys.ntu.edu.tw
   #SCM_PORT=6060
   SCM_HOST=dayabay.phys.ntu.edu.tw
   SCM_PORT=80
   
   SCM_USER=$USER
   SCM_PASS=$(private-val NON_SECURE_PASS)
   SCM_TRAC=env
   SCM_GROUP=NTU

   SCM_URL=http://$SCM_HOST

elif [ "$SCM_TAG" == "G" ]; then 

   ## trac testing 
   SCM_HOST=localhost
   SCM_PORT=80
   SCM_USER=$USER
   SCM_PASS=$(private-val NON_SECURE_PASS)
   SCM_TRAC=workflow
   SCM_GROUP=DEV
   
   SCM_URL=http://$SCM_HOST
   
else

   SCM_HOST=	
   SCM_PORT=	
   SCM_USER=
   SCM_PASS=
   SCM_GROUP=

fi


export SCM_URL
export SCM_HOST
export SCM_PORT
export SCM_GROUP
export SCM_TRAC





}
   
local-mode-pkgs(){ cat << EOL
tracpreq
python
apache
svn
sqlite
EOL
}
local-mode(){
  local msg="=== $FUNCNAME :"
  local tag=${1:-$NODE_TAG}
  echo $msg $tag 
  echo 
  func-
  local pkg
  local fmt="%-10s     %-20s %-20s %s \n"
  printf "$fmt" pkg default mode note 
  echo
  $FUNCNAME-pkgs | while read pkg ; do
     if [ "$(func-isfunc $pkg-)" == "y" ]; then 
        eval $pkg-
        local mode=$(eval $pkg-mode $tag)
        local default=$(eval $pkg-mode-default $tag)
        local note=""
        [ "$default"  != "$mode" ] && note="######"
        printf "$fmt" $pkg $default $mode  $note 

        #type $pkg-mode
        #type $pkg-mode-default
     else
        echo $msg ERROR $pkg- is not a function 
     fi
  done

  echo 
  echo the mode is influenced by the local environment _MODE variables ... ignore when checking other nodes 
  echo the default  is influenced by the NODE_TAG or argument ... allows to see defaults on other nodes
}

 
    
local-initialize(){

   local msg="=== $FUNCNAME :"
   local names="local-base local-base-env local-system-base local-scm-fold"
   echo $msg initializing dirs ... $names , check em with local-info
   local name
   for name in $names ; do 
      local dir=$(eval $name)
      echo $msg $name : $dir 
      if [ -d $dir ]; then 
         echo $msg $dir exists already 
      else 
         $SUDO mkdir -p $dir
         $SUDO chown $USER $dir
      fi
   done

   ## do here as it needs sudo pw too
   trac-inherit-setup

} 
    
