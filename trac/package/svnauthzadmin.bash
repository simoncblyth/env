svnauthzadmin-vi(){ vi $BASH_SOURCE ; }
svnauthzadmin-usage(){
   package-usage svnauthzadmin
   cat << EOU
 
       http://trac-hacks.org/wiki/SvnAuthzAdminPlugin

       Tickets ...
           * http://trac-hacks.org/query?status=!closed&component=SvnAuthzAdminPlugin&order=priority

       == experience ==

        * when adding a subj to a group 
           * the "subject" list is composed of all other groups and : *,admin,bitadmin,slave
               * why, what is special about these
               * these show up in the Trac permissions page ...
                   * http://dayabay.phys.ntu.edu.tw/tracs/env/admin/general/perm

       * after adding a new user via Accounts/Users 
           * the new user does not show up as an available subject to add to a group ? 
           * http://trac-hacks.org/ticket/5191 ... includes patch to get new users to show up
 
      == summary ===

        So the procedure to add a new user is :

         1) Accounts/Users               ... add the new user 
         2) Subversion/Subversion Access ....  pick a group and add the new user "subject" to it

       Without the users list patch there is an extra step ...

          1.5)  General/Permissions ... provide the new user with some trac permission eg WIKI_VIEW
                   *(you have to recall and retype the username) 


     == todo ==


 
EOU

}

svnauthzadmin-env(){
   elocal-
   package-
   trac- 
  local branch
  case $(trac-major) in 
     0.11) branch=0.11 ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export SVNAUTHZADMIN_BRANCH=$branch
}


svnauthzadmin-upgradeconf(){
   local msg="=== $FUNCNAME :"
   [ "$(trac-major)" != "0.11" ] && echo $msg this is only relevant to 0.11 && return 1
}


#svnauthzadmin-revision(){ echo 3768 ; }
svnauthzadmin-revision(){ echo 9290 ; }

svnauthzadmin-url(){     echo http://trac-hacks.org/svn/svnauthzadminplugin/$(svnauthzadmin-branch) ; }
svnauthzadmin-package(){ echo svnauthz; }

svnauthzadmin-fix(){
  local msg="=== $FUNCNAME :"
  local dir=$(svnauthzadmin-dir)
  echo $msg ... manual copying is replaced by the auto patching   
}

svnauthzadmin-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
svnauthzadmin-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
svnauthzadmin-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
svnauthzadmin-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
svnauthzadmin-get(){       package-get       ${FUNCNAME/-*/} $* ; }

svnauthzadmin-install(){   package-install   ${FUNCNAME/-*/} $* ; }
svnauthzadmin-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
svnauthzadmin-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
svnauthzadmin-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

svnauthzadmin-status(){    package-status    ${FUNCNAME/-*/} $* ; }
svnauthzadmin-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
svnauthzadmin-diff(){      package-diff      ${FUNCNAME/-*/} $* ; }
svnauthzadmin-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
svnauthzadmin-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

svnauthzadmin-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }
svnauthzadmin-update(){    package-fn $FUNCNAME $* ; }



svnauthzadmin-conf(){
   trac-
   trac-edit-ini $(trac-inipath $*) svnauthzadmin:show_all_repos:true

}



