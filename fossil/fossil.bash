# === func-gen- : fossil/fossil fgp fossil/fossil.bash fgn fossil fgh fossil
fossil-src(){      echo fossil/fossil.bash ; }
fossil-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fossil-src)} ; }
fossil-sdir(){     echo $(dirname $(fossil-source)) ; }
fossil-vi(){       vi $(fossil-source) $* ; }
fossil-env(){      
    elocal- 
    cfg- 
}
fossil-usage(){ cat << EOU
Fossil
========

Simple, high-reliability, distributed software configuration management

  * http://www.fossil-scm.org/fossil/doc/trunk/www/index.wiki

  * http://www.fossil-scm.org/download.html

      * release notes


Linux serving with xinetd 
--------------------------

After placing the config, need to restart the xinetd service::

	[blyth@cms01 e]$ curl http://localhost:591      ## just hangs
	[blyth@cms01 e]$ sudo /sbin/service xinetd status
	xinetd (pid 3099) is running...
	[blyth@cms01 e]$ sudo /sbin/service xinetd stop
	Stopping xinetd:                                           [  OK  ]
	[blyth@cms01 e]$ sudo /sbin/service xinetd start
	Starting xinetd:                                           [  OK  ]
	[blyth@cms01 e]$ curl http://localhost:591
	<h1>Not Found</h1>


Darwin serving with launchctl
------------------------------

::

    simon:fossil blyth$ sudo launchctl unload $(fossil-cfg-path)
    simon:fossil blyth$ sudo launchctl load $(fossil-cfg-path)



configure local options
-------------------------

::

    simon:fossil-src-20130216000435 blyth$ ./configure --help
      Local Options:               
      --with-openssl=path|auto|none  Look for openssl in the given path, or auto or none
      --with-zlib=path               Look for zlib in the given path
      --with-tcl=path                Enable Tcl integration, with Tcl in the specified path
      --with-tcl-stubs               Enable Tcl integration via stubs mechanism
      --disable-internal-sqlite      Don't use the internal sqlite, use the system one
      --static                       Link a static executable
      --disable-lineedit             Disable line editing
      --fossil-debug                 Build with fossil debugging enabled
      --json                         Build with fossil JSON API enabled
      --markdown                     Build with markdown engine enabled



EOU
}


#fossil-nam(){ echo fossil-src-20130216000435 ; }
#fossil-dir(){ echo $(local-base)/env/fossil/$(fossil-nam) ; }
fossil-nam(){ echo fossil ; }
fossil-dir(){ echo $HOME/$(fossil-nam) ; }

fossil-cd(){  cd $(fossil-dir); }
fossil-mate(){ mate $(fossil-dir) ; }
fossil-get(){
  echo now using the cloned trunk for latest fossil rather than tgz
}
fossil-get-from-tgz(){
   local dir=$(dirname $(fossil-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(fossil-nam)
   local tgz=$nam.tar.gz
   local url=http://www.fossil-scm.org/download/$tgz
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz
}


fossil-cd-build(){
   fossil-cd
   mkdir -p build
   cd build
}

fossil-build(){
   fossil-cd-build
   make clean
   ../configure
   make
}

fossil-build-custom(){
   fossil-cd-build
   make clean
   #../configure --json --markdown --fossil-debug
   ../configure --json --markdown 
   make
}
fossil-bin(){ echo $(fossil-dir)/build/fossil ; }
fossil-install(){ [ ! -x $(env-home)/bin/fossil ] &&  ln -s $(fossil-bin) $(env-home)/bin/fossil ; }




# server related funcs

fossil-cfg-path(){
   case $(uname) in 
     Darwin) echo /Library/LaunchDaemons/org.fossil-scm.fossil.plist ;;
     Linux)  echo /etc/xinet.d/fossil ;;
   esac    
}

# dont do that edit template OR config
#fossil-cfg-edit(){
#   local cmd="sudo vi $(fossil-cfg-path)"
#   echo $msg $cmd
#   eval $cmd
#}

fossil-tmpl(){ echo $(fossil-sdir)/$(basename $(fossil-cfg-path)).template ; } 

fossil-cfg-(){
   cfg-filltmpl- $(fossil-tmpl) fossil
}

fossil-cfg(){
   local msg=" == $FUNCNAME "
   local tgt=$(fossil-cfg-path)
   local tmp=/tmp/$FUNCNAME/$(basename $tgt) && mkdir -p $(dirname $tmp)
   fossil-cfg- > $tmp

   cat $tmp 
   local ans
   read -p "$msg write above filled template $tmp to target $tgt ? YES to proceed: " ans
   
   if [ "$ans" == "YES" ]; then  
       local cmd="sudo cp $tmp $tgt"
       echo $msg $cmd
       eval $cmd
   else
       echo $msg skipping
   fi
   rm $tmp
}

fossil-reload(){
   local cmd
   cmd=$(fossil-launchctl unload)
   echo $cmd
   eval $cmd
   cmd=$(fossil-launchctl load)
   echo $cmd
   eval $cmd
}

fossil-launchctl(){
   echo sudo launchctl $1 $(fossil-cfg-path)
}





# migration related

fossil-fromgit(){
    local msg=" === $FUNCNAME : "
    local name=$(basename $PWD)
    [ ! -d ".git" ]  && echo $msg this needs to run from a git repo top level directory which contains a .git directory  && return 1

    local ans
    read -p "$msg fast-export from git in $PWD into $name.fossil ? enter YES to proceed : " ans
    [ "$ans" != "YES" ] && echo $msg skipping && return 0

    date
    which git
    git --version
    which fossil
    fossil version
    git fast-export --all | fossil import --git $name.fossil
    date

}


