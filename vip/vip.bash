vip-src(){      echo vip/vip.bash ; }
vip-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vip-src)} ; }
vip-vi(){       vi $(vip-source) ; }

vip-usage(){
  cat << EOU

     References

          http://virtualenv.openplans.org/ 
          http://pip.openplans.org/
          http://packages.python.org/distribute/
               supported fork of setuptools 

          http://www.b-list.org/weblog/2008/dec/15/pip/
               advantages of pip over setuptools/easy_install etc..

     Typical vip workflow ... eg for npy (numpy- + mysql-python- experimentation on N)

          pip install -U virtualenv    
                               ## get uptodate virtualenv (eg 1.5.1 ) IN SYSTEM PYTHON

          vip-create npy       ## create a new environment 
          vip-activate npy     ## get into it 

          which python easy_install pip               
                               ## all these should be from the npy virtual env 

          which ipython nosetests         
                               ## not from npy virtual env

          pip install -U  ipython       ## plucked 0.10.1
          pip install -U  nose          ## plucked 0.11.1 N(py24) 0.11.4 C(py25)
          pip install -U  cython
                               ## are installed into system alreay hence the upgrade 
                               ## TODO : investigate standard essentials bootstrapping 

          pip -v install -e git://github.com/numpy/numpy.git#egg=numpy
                               ## need the latest numpy (~2.0?) for datetime support 
                         
          pip -v install -e svn+https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/matplotlib/#egg=matplotlib
                               ## need latest matplotlib (~1.0?) to work with latest numpy (~2.0?) 

          pip -v install -e hg+http://mysql-python.hg.sourceforge.net/hgweb/mysql-python/MySQLdb-2.0#egg=MySQLdb-2.0
                               ## need latest mysql-python- for mysqlmod.h to facilitate cython wrapping, earlier versions miss this

          pip install matplotlib              
                          ## plucked 0.91.1 ... this is ANCIENT AND A BUG ...  SHOULD BE 1.0.0

          pip install -U django       ## plucked 1.2.3 


                ... hmm will need flup + ... for deployment
                ... will django work with MySQLdb-2.0 ?? (ie 1.3.0 )
                       

          python -c "import sys ; print '\n'.join(sys.path) "
                               ## the npy virtual dirs should come ahead of the system ones   

     Numpy version checking ...

          (npy)[blyth@belle7 ~]$ python -c "import numpy as np ; print np.__version__ "
          2.0.0.dev-12d0200

          (npy)[blyth@cms01 v]$ python -c "import numpy as np ; print np.__version__ "
          2.0.0.dev-8fa2591
 
          (npy)[blyth@belle7 ~]$ pip install cython

   
     Initially found old system numpy shadowing the new npy one ???
          * resolved by updating virtualenv to 1.5.1 (sudo pip install --upgrade virtualenv) in system python
          * the 1.5.1 created virtual env comes with pip + setuptools(easy_install) 


     vip- 
          a combination of the triplet : virtualenv + pip + setuptools 
          provides python package assembly based on frozen versionsets 
          into virtual python environments  

          the use of virtual python environments means that once the triplet 
          are installed into the system (or base) python (probably 
          requiring sudo access) subsequent installations can be done without 
          sudo 

     vip-src : $(vip-src)
     vip-dir : $(vip-dir)


     vip-ls  :
         list virtual python environments     

     vip-preqs
         check the versions of :  setuptools / virtualenv / pip 
         this triplet is tightly coupled and other plvdbi constraints demand setuptools 0.6c11
         so using the precise versions is a necessity
      
     vip-create name 
         create virtual python environment dir  

     vip-check name
         check the virual python env and activate it


     vip-install <name>

        Usage : 
           cat production.pip | vip-install dbi
  
          Background, pip options : 
            -s  :  use site packages when creating v : eg for MySQLdb that needs to come from system
                   (DOES NOT WORK ... SO CREATE virtualenv SEPARATELY)

      --no-deps :  so the requirements file is fully in control of versions 
       -E \$dir :  v directory to use OR create
       -r \$pip :  requirements file specifying the versions to use
    --src \$src : directory to check out/clone editables into, avoid repeating slow clones      

      the srcdir is located one step above the individual env allowing editable paths  :
            -e ../src/


      TODO:


EOU
}


vip-base(){ echo $(local-base)/env/v ; }  ## the v should be "vip"
#vip-name(){ echo ${1:-$VIP_NAME} ; }
vip-name(){ basename ${VIRTUAL_ENV:-${1:-.}} ; } 
vip-dir(){  echo $(vip-base)/$(vip-name $1) ; }
vip-srcdir(){  echo $(vip-base)/src ; }
vip-cd(){   cd $(vip-dir $*); }
vip-mate(){ mate $(vip-dir $*) ; }
vip-activate(){    . $(vip-dir $*)/bin/activate ;  }
vip-reqpath(){ echo $(vip-dir $*)/requirememts.txt ; }
vip-deactivate(){  deactivate ; }

vip-dbi(){  vip-activate dbi ; }
vip-npy(){  vip-activate npy ; }

vip-env(){ echo -n ; }
vip--(){ 
   local msg="=== $FUNCNAME :"
   local cmd="pip -E $(vip-dir) install --src=$(vip-srcdir) --no-deps $*"
   echo $msg \"$cmd\"
   eval $cmd
}


vip-setuptools-version(){ python -c "import setuptools ; print setuptools.__version__ " ; }
vip-pip-version-(){    pip --version ; }
vip-second(){          echo -n $2 ; }
vip-pip-version(){     vip-second $($FUNCNAME-) ; } 


vip-preqs(){

 local msg="=== $FUNCNAME :" 

 echo $msg if version mismatches, try to install into the target python, currently : $(which python)  

 local rc=0
 local vsv=$(vip-setuptools-version) 
 local esv="0.6c11"
 local msv="setuptools version $vsv is not the required $esv ... try : sudo easy_install -U setuptools "  
 [ "$vsv" != "$esv" ] && echo $msg $msv && rc=1

 local vev=$(virtualenv --version)
 local eev="1.3.4dev"   ## actually needs to be > some specifc revision that supports 0.6c11 ... this version does not ensure that 
 local mev="virtualenv version $vev is not the required $eev ... try : sudo easy_install -U virtualenv==dev :  dev version needed for 0.6c11 compatibility "
 [ "$vev" != "$eev" ] && echo $msg $mev && rc=2

 local vpv=$(vip-pip-version)
 local epv="0.5.1"
 local mpv="pip version $vpv is not the required $epv ... try : sudo easy_install -U pip==dev : dev version needed for uninstall functionality "
 [ "$vpv" != "$epv" ] && echo $msg $mpv && rc=3
 echo $msg suspect pip version reporting broken ... $mpv

 #return $rc   ... try with the more recent 
 return 0
}


vip-create(){
   local msg="=== $FUNCNAME :"
   local name=${1:-dbi}
   local vdir=$(vip-dir $name)

   [ -d "$vdir" ] && echo $msg virtual python environment $vdir exists already && return 0
   echo $msg creating $name virtual python environment in $vdir
   local dir=$(dirname $vdir) &&  mkdir -p $dir && cd $dir
   local msg="=== $FUNCNAME :"
   [ "$(which virtualenv)" == "" ] && echo $msg missing virtualenv && return 1
   [ "$(which pip)" == "" ] && echo $msg missing pip && return 1
   [ "$(which hg)" == "" ] && echo $msg missing hg && return 1

   virtualenv $vdir
}

vip-check(){
   local msg="=== $FUNCNAME :"
   local name=${1:-dbi}
   local vdir=$(vip-dir $name)

   ## get into the virtial python environment 
   vip-activate $name

   [ "$(which python)" != "$vdir/bin/python" ] && echo $msg ABORT failed to setup virtual python $(which python)   && return 1
   python -c "import MySQLdb" 2> /dev/null
   [ ! $? -eq 0 ] && echo $msg ABORT missing MySQLdb ... see pymysql-build or system install if using system python  && return 1
   return 0

}





vip-ls(){ ls -l $(vip-base) ; }


vip-install(){
  local msg="=== $FUNCNAME :"
  local nam=$(vip-name $*)
  shift
  local cmd
  local sdir=$(vip-srcdir)
  [ ! -d "$sdir" ] && mkdir -p $sdir
  local dir=$(vip-dir $nam)
  local req=$(vip-reqpath $nam)   
  [ ! -d "$dir" ] && mkdir -p $dir

  cmd="cd $dir"
  echo $msg \"$cmd\"
  eval $cmd

  cmd="virtualenv $dir "
  echo $msg \"$cmd\"
  eval $cmd

  echo $msg stream stdin into the requirements $req
  cat - > $req

  cmd="pip -E $dir install --no-deps -r $req --src=$sdir --log=$dir/pip.log   $* "
  echo $msg installation into $dir ... based on the requirements : $req
  echo $msg \"$cmd\"  
  eval $cmd
}




vip-freeze(){
  local msg="=== $FUNCNAME :"
  local msg="=== $FUNCNAME :"
  local nam=$(vip-name $*)
  local dir=$(vip-dir $nam)
  shift

  local req=$(vip-reqpath $nam)   
  echo $msg nam $nam dir $dir req $req
  local tmp=/tmp/env/$FUNCNAME/$nam/$(basename $req) && mkdir -p $(dirname $tmp)
  local cmd
  if [ -f "$req" ] ; then
     cmd="pip -E $dir freeze -r $req "   ## -r 
  else
     cmd="pip -E $dir freeze "
  fi
  echo $msg \"$cmd\"
  echo $msg freezing the state of python into $tmp ... for possible updating of $req
  eval $cmd > $tmp

  if [ -f "$req" ]; then 
     diff $req $tmp
     echo $msg NOT COPYING AS TOO MESSY FOR AUTOMATION ... DO THAT YOURSELF WITH : \"cp $tmp $req\"
  else
     echo $msg copying initial pip freeze to $req
     cp $tmp $req
  fi
}






vip-build(){
   #vip-get 
   vip-preqs
   [ ! $? -eq 0 ]  && return 1
   env-build
}

vip-wipe(){
   local msg="=== $FUNCNAME :"
   local cmd="rm -rf $(vip-dir) " 
   local ans
   read -p "$msg proceed with \"$cmd\"  enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg skipped && return 0
   eval $cmd
}

vip-get-deprecated(){
   local dir=$(dirname $(vip-dir)) &&  mkdir -p $dir && cd $dir
   local msg="=== $FUNCNAME :"
   [ "$(which virtualenv)" == "" ] && echo $msg missing virtualenv && return 1
   [ "$(which pip)" == "" ] && echo $msg missing pip && return 1
   [ "$(which hg)" == "" ] && echo $msg missing hg && return 1
   [ ! -d "$(vip-dir)" ] && virtualenv $(vip-dir) || echo $msg virtualenv dir $(vip-dir) exists already skipping virtualenv creation 
   ! vip-activate && echo "$msg failed to activate " && return 1 
   [ "$(which python)" != "$(vip-dir)/bin/python" ] && echo $msg ABORT failed to setup virtual python $(which python)   && return 1
   python -c "import MySQLdb" 2> /dev/null
   [ ! $? -eq 0 ] && echo $msg ABORT missing MySQLdb ... see pymysql-build or system install if using system python  && return 1
   return 0
}

## handle such node specifics in .bash_profile 
vip-env-C(){
   ## system python on cms01 : 2.3.4 is too old to use ... so the basis of the virtualenv is the source python
   python- source 
}

vip-env-deprecated(){      
   elocal- ; 
   case $NODE_TAG in  
     C) vip-env-C ;;
   esac
  
   #vip-activate ;  
   ## this distinguishes deployed running and debug running 
   #case $(vip-mode) in 
   #  dev) export ENV_PRIVATE_PATH=$HOME/.bash_private ;; 
   #    *) export ENV_PRIVATE_PATH=$(apache-private-path) ;;
   #esac
}
