tractags-usage(){
   plugins-usage tractags
   cat << EOU
# http://www.trac-hacks.org/wiki/TagsPlugin
   
   tractags-fix :
       copy in changed files sourced in env for safety
     ... this is intended for minor customizations, anything major should be housed in tracdev repo
 
EOU

}

tractags-env(){
   elocal-
   tplugins-
   
   #export TRACTAGS_BRANCH=tags/0.4.1
   #export TRACTAGS_BRANCH=tags/0.5
   #export TRACTAGS_BRANCH=tags/0.6
   #export TRACTAGS_BRANCH=trunk
   
   export TRACTAGS_BRANCH=trunk_cust
   
   # using names like trunk-cust ... mysteriously result in egg names with trunk_cust 
   #  ... so start with the underscored
}

tractags-url(){     echo http://trac-hacks.org/svn/tagsplugin/$(tractags-obranch) ; }
tractags-module(){  echo tractags ; }
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
  cd $ENV_HOME/trac/plugins/tractags  
  #cp setup.py  $dir/
  cp macros.py $dir/tractags/

}

tractags-obranch(){   plugins-obranch   ${FUNCNAME/-*/} $* ; }
tractags-branch(){    plugins-branch    ${FUNCNAME/-*/} $* ; }
tractags-basename(){  plugins-basename  ${FUNCNAME/-*/} $* ; }
tractags-dir(){       plugins-dir       ${FUNCNAME/-*/} $* ; } 
tractags-egg(){       plugins-egg       ${FUNCNAME/-*/} $* ; }
tractags-get(){       plugins-get       ${FUNCNAME/-*/} $* ; }
tractags-cust(){      plugins-cust      ${FUNCNAME/-*/} $* ; }
tractags-install(){   plugins-install   ${FUNCNAME/-*/} $* ; }
tractags-uninstall(){ plugins-uninstall ${FUNCNAME/-*/} $* ; }
tractags-reinstall(){ plugins-reinstall ${FUNCNAME/-*/} $* ; }
tractags-enable(){    plugins-enable    ${FUNCNAME/-*/} $* ; }


tractags-diff(){
    local dir=$(tractags-dir)
    cd $(dirname $dir)    
    diff -r --brief $TRACTAGS_BASENAME TracTags-trunk | grep -v .svn
}

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



tractags-env-upgrade(){
    local name=${1:-$SCM_TRAC}
    local env=$SCM_FOLD/tracs/$name
    sudo trac-admin $env upgrade
}





