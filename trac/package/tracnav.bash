
tracnav-usage(){

   package-usage ${FUNCNAME/-*/}
   cat << EOU
EOU

}

tracnav-env(){
   elocal-
   python-

   export TRACNAV_BRANCH=tracnav-0.11_cust
   #export TRACNAV_BRANCH=tracnav
}

tracnav-docurl(){   echo http:// ; }
tracnav-url(){      echo http://svn.ipd.uka.de/repos/javaparty/JP/trac/plugins/$(tracnav-obranch) ;}
tracnav-package(){  echo tracnav ; }
tracnav-eggbas(){   echo TracNav ; }

tracnav-eggver(){
    local ob=$(tracnav-obranch)
    local v
    case $ob in 
          tracnav-0.11) v=4.0pre6 ;;
               tracnav) v=3.92    ;;
                     *) v=$ob ;;
    esac
    echo $v
}


tracnav-fix(){
   local msg="=== $FUNCNAME :"
   echo $msg no fix needed
}



tracnav-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
tracnav-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tracnav-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tracnav-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tracnav-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tracnav-get(){       package-get       ${FUNCNAME/-*/} $* ; }
tracnav-cust(){      package-cust      ${FUNCNAME/-*/} $* ; }
tracnav-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tracnav-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tracnav-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tracnav-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tracnav-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tracnav-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tracnav-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
tracnav-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
tracnav-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }