tractrac-src(){    echo trac/package/tractrac.bash  ; }
tractrac-source(){ echo ${BASH_SOURCE:-$(env-home)/$(tractrac-src)} ; }
tractrac-vi(){     vi $(tractrac-source) ; }

tractrac-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
    addendum to tractrac-install
         In addition to installing the package into PYTHON_SITE this 
         also installs the trac-admin and tracd entry points by default 
         to /usr/local/bin/{trac-admin,tracd}  
         
         
    tractrac-branch2revision :
         note the revision moniker on cms01 was formerly incorrectly 
         the head revision at the time of installation 7326, 
         should use the revision at creation of the tag 
         ...  for patch name matching between machines
         NB makes no difference to the actual code, but would prevent patches
         from being found
         


  == operational notes ==

    Very large diffs bring Trac/apache to knees ... need

     * http://trac.edgewall.org/ticket/2343

   The default of 10 million bytes is too much ...
{{{
[changeset]
max_diff_bytes = 10000000
max_diff_files = 0
}}}

  at IHEP got Q to djust limit to 100000 bytes (down factor of 100 from 10 million)


            
         
         
         
EOU

}

tractrac-env(){
  elocal-
  package-
  
  export TRACTRAC_BRANCH=$(tractrac-version2branch $TRAC_VERSION)
}

tractrac-version2branch(){

  ## http://trac.edgewall.org/browser/tags
  case $1 in 
       trunk) echo trunk ;;
      0.11.1) echo tags/trac-0.11.1  ;;
        0.11) echo tags/trac-0.11    ;;
     0.11rc1) echo tags/trac-0.11rc1 ;; 
      0.11b1) echo tags/trac-0.11b1  ;;
      0.10.4) echo tags/trac-0.10.4  ;;
      0.11.4) echo tags/trac-0.11.4  ;;
  esac
}

tractrac-branch2revision(){
   case $1 in 
      tags/trac-0.11) echo 7236 ;; 
    tags/trac-0.11.1) echo 7451 ;;    
                   *) echo HEAD ;;
   esac
}


tractrac-findit-(){ python -c "import trac as _ ; print _.__file__" ;  }
tractrac-findit(){  local init=$($FUNCNAME-) ; [ -n "$init" ] && echo $(dirname $init) ; }

tractrac-cssfix(){
  local msg="=== $FUNCNAME :"
  local css=$(python-site)/$(tractrac-egg)/trac/htdocs/css/ticket.css
  if [ ! -f "$css" ]; then
     css=$(tractrac-findit)/htdocs/css/ticket.css
  fi 
  [ ! -f "$css" ] && echo $msg ABORT no css $css && return 1
  echo $msg found css $css

  local tmp=/tmp/env/$FUNCNAME/$(basename $css)

  echo $msg $css
  mkdir -p $(dirname $tmp)
  perl -p -e 's,(\#content.ticket.*width: )(\d+)(px.*)$,${1}1201${3},' $css > $tmp

  diff $css $tmp
  local ans
  read -p "$msg change $css as proposed ? enter YES to proceed " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 0

  local cmd="sudo cp $tmp $css"
  echo $cmd
  eval $cmd
}



tractrac-revision(){
   echo $(tractrac-branch2revision $(tractrac-version2branch $TRAC_VERSION))
}

tractrac-url(){     echo http://svn.edgewall.org/repos/trac/$(tractrac-branch) ;}
tractrac-pkgname(){ echo trac ; }

tractrac-fix(){
   cd $(tractrac-dir)   
   echo no fixes
}

tractrac-makepatch(){  package-fn $FUNCNAME $* ; }
tractrac-applypatch(){ package-fn $FUNCNAME $* ; }






tractrac-branch(){    package-fn  $FUNCNAME $* ; }
tractrac-basename(){  package-fn  $FUNCNAME $* ; }
tractrac-dir(){       package-fn  $FUNCNAME $* ; }  
tractrac-egg(){       package-fn  $FUNCNAME $* ; }
tractrac-get(){       package-fn  $FUNCNAME $* ; }    

tractrac-install(){   package-fn  $FUNCNAME $* ; }
tractrac-uninstall(){ package-fn  $FUNCNAME $* ; } 
tractrac-reinstall(){ package-fn  $FUNCNAME $* ; }
tractrac-enable(){    package-fn  $FUNCNAME $* ; }  

tractrac-status(){    package-fn  $FUNCNAME $* ; } 
tractrac-auto(){      package-fn  $FUNCNAME $* ; } 
tractrac-diff(){      package-fn  $FUNCNAME $* ; } 
tractrac-rev(){       package-fn  $FUNCNAME $* ; } 
tractrac-cd(){        package-fn  $FUNCNAME $* ; } 

tractrac-fullname(){  package-fn  $FUNCNAME $* ; } 
tractrac-update(){    package-fn  $FUNCNAME $* ; } 







