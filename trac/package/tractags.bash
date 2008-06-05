tractags-usage(){
   package-usage tractags
   cat << EOU
# http://www.trac-hacks.org/wiki/TagsPlugin
   
   tractags-fix :
       copy in changed files sourced in env for safety
     ... this is intended for minor customizations, anything major should be housed in tracdev repo
 
EOU

}

tractags-env(){
   elocal-
   tpackage-
   
   #export TRACTAGS_BRANCH=tags/0.4.1
   #export TRACTAGS_BRANCH=tags/0.5
   #export TRACTAGS_BRANCH=tags/0.6
   #export TRACTAGS_BRANCH=trunk
   
   export TRACTAGS_BRANCH=trunk_cust
   
   # using names like trunk-cust ... mysteriously result in egg names with trunk_cust 
   #  ... so start with the underscored
}

tractags-url(){     echo http://trac-hacks.org/svn/tagsplugin/$(tractags-obranch) ; }
tractags-package(){ echo tractags ; }
tractags-eggbas(){  echo TracTags ; }

tractags-eggver(){
    local ob=$(tractags-obranch)
    case $ob in 
       trunk) echo 0.6 ;;
           *) echo $ob ;;
    esac
}

tractags-fix(){
  local dir=$(tractags-dir)
  cd $ENV_HOME/trac/package/tractags  
  #cp setup.py  $dir/
  cp macros.py $dir/tractags/

}

tractags-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
tractags-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tractags-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tractags-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tractags-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tractags-get(){       package-get       ${FUNCNAME/-*/} $* ; }
tractags-cust(){      package-cust      ${FUNCNAME/-*/} $* ; }
tractags-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tractags-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tractags-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tractags-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tractags-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tractags-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tractags-diff(){      package-diff      ${FUNCNAME/-*/} $* ; }
tractags-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
tractags-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }


tractags-unconf(){

   local msg="=== $FUNCNAME :"
   local name=${1:-$SCM_TRAC}
   local tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   local ver=$(basename $TRACTAGS_BRANCH)
   
   if [ "$ver" == "0.6" -o "$ver" == "trunk" ]; then
      echo $msg this is only relevant to pre 0.6 versions
   else
      trac-ini-
      trac-ini-edit $tini trac:default_handler:WikiModule
   fi

}

tractags-conf(){
  
   local msg="=== $FUNCNAME :"
   local name=${1:-$SCM_TRAC}
   
   local tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   
    if [ "$TRACTAGS_BRANCH" == "tags/0.5" ]; then
   
       echo $msg CAUTION changing the trac:default_handler:TagsWikiModule from WikiModule
       echo $msg this set up is for version 0.4 or 0.5 of tractags ... not the 0.6 version that is needed with 0.11 of trac 
       echo $msg see http://trac-hacks.org/wiki/TagsPlugin/0.5/Installation 
   
       trac-ini- 
       trac-ini-edit $tini trac:default_handler:TagsWikiModule components:trac.wiki.web_ui.wikimodule:disabled components:tractags.\*:enabled

    elif [ "$TRACTAGS_BRANCH" == "tags/0.6" -o "$TRACTAGS_BRANCH" == "trunk" ]; then
    
        trac-ini-
        trac-ini-edit $tini components:tractags.\*:enabled
    
    else
    
        echo $msg ERROR branch $TRACTAGS_BRANCH not handled
    fi
}








