


trac-plugin-webadmin-get(){

  cd $LOCAL_BASE/trac
  mkdir -p plugins && cd plugins
  svn co http://svn.edgewall.com/repos/trac/sandbox/webadmin/

}

trac-plugin-webadmin-install(){

  cd $LOCAL_BASE/trac/plugins 
  cd webadmin
  python setup.py install

}

trac-plugin-webadmin-enable(){
   
   name=${1:-$SCM_TRAC}
   ini-edit $SCM_FOLD/tracs/$name/conf/trac.ini components:webadmin.\*:enabled
}

