
tracnav-usage(){

   plugins-usage ${FUNCNAME/-*/}
   cat << EOU
EOU

}

tracnav-env(){
   elocal-
   python-

   export TRACNAV_BRANCH=tracnav-0.11
   #export TRACNAV_BRANCH=tracnav
}


tracnav-url(){      echo http://svn.ipd.uka.de/repos/javaparty/JP/trac/plugins/$(tracnav-obranch) ;}
tracnav-module(){   echo tracnav ; }
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



tracnav-obranch(){   plugins-obranch   ${FUNCNAME/-*/} $* ; }
tracnav-branch(){    plugins-branch    ${FUNCNAME/-*/} $* ; }
tracnav-basename(){  plugins-basename  ${FUNCNAME/-*/} $* ; }
tracnav-dir(){       plugins-dir       ${FUNCNAME/-*/} $* ; } 
tracnav-egg(){       plugins-egg       ${FUNCNAME/-*/} $* ; }
tracnav-get(){       plugins-get       ${FUNCNAME/-*/} $* ; }
tracnav-cust(){      plugins-cust      ${FUNCNAME/-*/} $* ; }
tracnav-install(){   plugins-install   ${FUNCNAME/-*/} $* ; }
tracnav-uninstall(){ plugins-uninstall ${FUNCNAME/-*/} $* ; }
tracnav-reinstall(){ plugins-reinstall ${FUNCNAME/-*/} $* ; }
tracnav-enable(){    plugins-enable    ${FUNCNAME/-*/} $* ; }



 