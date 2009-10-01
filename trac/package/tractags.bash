tractags-vi(){ vi $BASH_SOURCE ; }
tractags-usage(){
   package-usage tractags
   cat << EOU
# http://www.trac-hacks.org/wiki/TagsPlugin
   
   tractags-fix :
       copy in changed files sourced in env for safety
     ... this is intended for minor customizations, anything major should be housed in tracdev repo
  

   Simple fix for IE TagCloud rendering all on one line ... 
      http://trac-hacks.org/ticket/4223

[blyth@cms02 tractags]$ vi macros.py
[blyth@cms02 tractags]$ pwd
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/TracTags-0.6-py2.5.egg/tractags
 
--- macros.py.orig      2009-02-18 10:11:22.000000000 +0100
+++ macros.py   2009-02-18 08:46:20.000000000 +0100
@@ -50,7 +50,7 @@
         if i == last:
             li(class_='last')
         li()
-        ul(li)
+        ul(li, ' ')
     return ul


  
  
EOU

}

tractags-env(){
   elocal-
   package-
   
   trac- 
    
  local branch
  case $(trac-major) in 
     0.10) branch=tags/0.6  ;;
     0.11) branch=trunk     ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export TRACTAGS_BRANCH=$branch
   
}


tractags-upgradeconf(){

   local msg="=== $FUNCNAME :"
   [ "$(trac-major)" != "0.11" ] && echo $msg this is only relevant to 0.11 && return 1

   

}


#tractags-revision(){ echo 3768 ; }
tractags-revision(){ echo 3882 ; }

tractags-url(){     echo http://trac-hacks.org/svn/tagsplugin/$(tractags-branch) ; }
tractags-package(){ echo tractags ; }

tractags-fix(){
  local msg="=== $FUNCNAME :"
  local dir=$(tractags-dir)
  #cd $ENV_HOME/trac/package/tractags  
  #cp setup.py  $dir/
  #cp macros.py $dir/tractags/

  echo $msg ... manual copying is replaced by the auto patching   
}

tractags-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tractags-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tractags-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tractags-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tractags-get(){       package-get       ${FUNCNAME/-*/} $* ; }

tractags-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tractags-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tractags-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tractags-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tractags-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tractags-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tractags-diff(){      package-diff      ${FUNCNAME/-*/} $* ; }
tractags-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
tractags-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

tractags-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }
tractags-update(){    package-fn $FUNCNAME $* ; }

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








