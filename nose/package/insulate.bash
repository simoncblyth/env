insulate-vi(){ vi $BASH_SOURCE ; }
insulate-usage(){
   package-fn  $FUNCNAME $*
   cat << EOU
   
      http://code.google.com/p/insulatenoseplugin/wiki/Documentation


       nosetests -v --with-insulate --insulate-every-test 

 
EOU
}

insulate-env(){ elocal- ; }
insulate-rev(){       echo 38 ; }
insulate-patchpath(){ echo $(env-home)/trac/patch/insulate/insulate-trunk-$(insulate-rev).patch ; }
insulate-url(){       echo http://insulatenoseplugin.googlecode.com/svn/trunk ; }
insulate-name(){      echo insulatenoseplugin ; }

insulate-dir(){       echo $(local-base)/env/nose/$(insulate-name) ; }
insulate-srcdir(){    echo $(insulate-dir)/insulate ; }
insulate-getdir(){    echo $(dirname $(dirname $(insulate-srcdir))) ; }

insulate-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(insulate-getdir) 
   mkdir -p $dir && cd $dir
   local cmd="svn checkout $(insulate-url)@$(insulate-rev) $(insulate-name) "
   echo $msg $cmd ... from $PWD
   eval $cmd
   local pp=$(insulate-patchpath)
   [ ! -f "$pp" ] && echo $msg no patch $pp && return 0
   cd $(insulate-dir)
   patch -p0 < $pp
}

insulate-install(){ 

   # just a link is not sufficient for nosetests -p 
   #python-
   #python-ln $(insulate-srcdir)  

  cd $(insulate-dir)
  python setup.py develop
}

insulate-ez(){ easy_install -Z InsulateRunner ; }
insulate-chk(){
  python -c "import insulate as _ ; print _.__file__ "
}
