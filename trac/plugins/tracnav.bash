
tracnav-usage(){

   cat << EOU

      http://svn.ipd.uka.de/trac/javaparty/wiki/TracNav
   
    tracnav- precursor is defined in trac/plugins/plugins.bash via tplugins- precursor
    
    Functions of TRACNAV_BRANCH : $TRACNAV_BRANCH
    use a branch name with _cust appended for local customizing 
    
    tracnav-basename : $(tracnav-basename)
    tracnav-url      :  $(tracnav-url)
    tracnav-dir      :  $(tracnav-dir)
    tracnav-egg      :  $(tracnav-egg)
        NB IN _cust CASE THIS FUNC AND setup.py need to be edited in parallel


    tracnav-get
          svn co the -url into the -dir

    tracnav-install  :
           easy install into PYTHON_SITE $PYTHON_SITE
           
    tracnav-uninstall :
           remove the -egg and easy-install.pth reference
           
    tracnav-reinstall :
        uninstall then reinstall ... eg to propagate customizations 


    Usage :
        tplugins-
        tracnav-usage

     Get a branch ready for customization 
        TRACNAV_BRANCH=tracnav-0.11_cust tracnav-get

     Check the dynamics 
         TRACNAV_BRANCH=tracnav-0.11_cust tracnav-usage


EOU

}

tracnav-env(){
   elocal-
   python-

   export TRACNAV_BRANCH=tracnav-0.11
   #export TRACNAV_BRANCH=tracnav
}

tracnav-basename(){ echo $TRACNAV_BRANCH ; }
tracnav-dir(){      echo $LOCAL_BASE/env/trac/plugins/tracnav/$(tracnav-basename) ;}
tracnav-url(){      
   local b=$TRACNAV_BRANCH  
   ## if the branch ends with _cust then strip this in forming the url   
   [ "${b:$((${#b}-5))}" == "_cust" ] && b=${b/_cust/} 
   echo http://svn.ipd.uka.de/repos/javaparty/JP/trac/plugins/$b 
}
tracnav-egg(){      
   local b=$TRACNAV_BRANCH
   local eggver
   case $b in 
      tracnav-0.11) eggver=4.0pre6 ;;
 tracnav-0.11_cust) eggver=4.0pre6_cust ;;
           tracnav) eggver=3.92    ;;
                 *) eggver=$b      ;; 
   esac
   echo TracNav-$eggver-py2.5.egg
}


tracnav-get(){
   local dir=$(tracnav-dir)
   local pir=$(dirname $dir)
   mkdir -p $pir
   cd $pir   
   svn co $(tracnav-url) $(tracnav-basename) 
}

tracnav-install(){
   cd $(tracnav-dir)
   $SUDO easy_install -Z .   
}

tracnav-uninstall(){
   python-uninstall $(tracnav-egg)
}

tracnav-reinstall(){
  tracnav-uninstall $*
  tracnav-install $*
}


tracnav-enable(){
   local name=${1:-$SCM_TRAC}
   trac-ini-
   trac-ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:tracnav.\*:enabled
}