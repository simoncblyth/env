tractags-usage(){

   cat << EOU

   Precursor "tractags-" is defined in scm/trac/trac.bash with precursor "trac-"


  # http://www.trac-hacks.org/wiki/TagsPlugin

   TRACTAGS_BRANCH : $TRACTAGS_BRANCH
   TRACTAGS_EGG    : $TRACTAGS_EGG
   PYTHON_SITE     : $PYTHON_SITE
   
   tractags-url    : $(tractags-url) 
   tractags-dir    : $(tractags-dir)


   tractags-get  :  
        checkout from $(tractags-url) into $(tractags-dir)

 
   Usage :
       trac-
       tractags-
       tractags-usage

   Get the default version and the trunk version for comparison :  
       tractags-get
       TRACTAGS_BRANCH=trunk tractags-get

   Uninstall the default version, and install the trunk version in its place
   
       tractags-uninstall
       TRACTAGS_BRANCH=trunk tractags-install  



EOU

}


tractags-ls(){

   ls -l $PYTHON_SITE/

}


tractags-diff(){

    local dir=$(tractags-dir)
    cd $(dirname $dir)
    
    diff -r --brief $TRACTAGS_BASENAME TracTags-trunk | grep -v .svn

}



tractags-uninstall(){
   python-uninstall $TRACTAGS_EGG
}


tractags-env(){
   elocal-
   python-
   
   export TRACTAGS_NAME=TracTags
   
   #export TRACTAGS_BRANCH=tags/0.4.1
   #export TRACTAGS_BRANCH=tags/0.5
   #export TRACTAGS_BRANCH=tags/0.6
   #export TRACTAGS_BRANCH=trunk
   export TRACTAGS_BRANCH=trunk-cust
   
   export TRACTAGS_BASENAME=$TRACTAGS_NAME-$(basename $TRACTAGS_BRANCH)
   
   local ver=$(basename $TRACTAGS_BRANCH)
   
   #
   # the name of the egg depends on what is in the setup.py ... 
   # ... it does not correspond to the branch ... although it should do
   #
   
   local eggver
   case $ver in
      trunk) eggver=0.6 ;;
 trunk-cust) eggver=trunk-cust ;;
          *) eggver=$ver ;;
    esac
    
    export TRACTAGS_EGG=${TRACTAGS_NAME}-$eggver-py2.5.egg
   
}

tractags-dir(){ echo $LOCAL_BASE/env/trac/plugins/${TRACTAGS_BASENAME} ; }
tractags-url(){ 

  local branch=$TRACTAGS_BRANCH 
  echo http://trac-hacks.org/svn/tagsplugin/$branch 
   
}


tractags-get(){
   local dir=$(tractags-dir)
   mkdir -p $(dirname $dir)
   cd $(dirname $dir)   
   local url=$(tractags-url)  
   svn co $url  $(basename $dir) 
}


tractags-install(){
      
   cd $(tractags-dir)
   $SUDO easy_install -Z .

}


#  argh ... the name of the egg depends on whats in the setup.py ...
# in this case the trunk version still has 0.6 
#
# Running setup.py -q bdist_egg --dist-dir /usr/local/env/trac/plugins/TracTags-trunk/egg-dist-tmp-HAe2ER
# zip_safe flag not set; analyzing archive contents...
# Adding TracTags 0.6 to easy-install.pth file
#
# Installed /Library/Python/2.5/site-packages/TracTags-0.6-py2.5.egg
# Processing dependencies for TracTags==0.6
# Finished processing dependencies for TracTags==0.6
#
#




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





