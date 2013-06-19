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
Fossil SCM
===========

.. contents:: :local:

Simple, high-reliability, distributed software configuration management

* http://www.fossil-scm.org/fossil/doc/trunk/www/index.wiki
* http://www.fossil-scm.org/download.html


FOSSIL URLs
------------

* http://localhost:591/env/timeline


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

Alternatively the function *fossil-reload* does this


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


Functions
-----------

Operational Functions
~~~~~~~~~~~~~~~~~~~~~~~

*fossil-url-check [url]*
       check fossil is responding::

            simon:e blyth$ fossil-url-check                     
            === fossil-url-check : INFO fossil running OK at http://localhost:591
            simon:e blyth$ fossil-url-check http://localhost:592
            === fossil-url-check : WARN FOSSIL NOT RUNNING AT http://localhost:592

Build related functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

*fossil-get*
       download and unpack the tarball distribution

*fossil-cd-build*
       cd to the fossil build directory, creating it if it doesnt exist

*fossil-build*
       build with plain vanilla config 

*fossil-build-custom*
       build with non default config options such as markdown and json support

*fossil-bin*
       location of fossil binary in the build directory

*fossil-install*
       plan symbolic link to binary 

Config Functions
~~~~~~~~~~~~~~~~~~~

*fossil-cfg-path*
       path to the config file

*fossil-cfg-edit*
       direct editing the config to test changes before persisting them in the template 

*fossil-tmpl*
       path to the config template

*fossil-cfg-*
       fill the template and emit to stdout for checking, the context used for filling comes
       from the *[fossil]* section of the ini config file *~/.env.cnf*

*fossil-cfg*
       uses *fossil-cfg-* to generate config and copy it into the needed location

*fossil-reload*
       *OSX specific* : uses launchctl to unload, then load the fossil config

*fossil-launchctl cmd*
       launchctl *cmd* operations on the *fossil-cfg-path*


Migration Functions
~~~~~~~~~~~~~~~~~~~~~

*fossil-fromgit*
       exploring migrations from git into fossil


EOU
}


fossil-name(){ echo fossil-src-20130216000435 ; }
fossil-dir(){ echo $(local-base)/env/fossil/$(fossil-name) ; }
fossil-cd(){  cd $(fossil-dir); }
fossil-mate(){ mate $(fossil-dir) ; } 
fossil-get(){
   local dir=$(dirname $(fossil-dir)) &&  mkdir -p $dir && cd $dir
   local name=$(fossil-name)
   local tgz=$name.tar.gz
   local url=http://www.fossil-scm.org/download/$tgz
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$name" ] && tar zxvf "$tgz"
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

fossil-cfg-edit(){
   echo $msg WARNING : changes will be overriddedn by template filling : use only to test new config prior to altering template OR config 
   local cmd="sudo vi $(fossil-cfg-path)"
   echo $msg $cmd
   eval $cmd
}

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

   echo $msg remember to fossil-reload to act upon the change now

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

fossil-url(){
   echo http://localhost:591
   # caution duplication of port defined in ~/.env.cnf [fossil]
}

fossil-url-check(){
   local msg=" === $FUNCNAME : "
   local url=${1:-$(fossil-url)}
   [ "$(curl -s $url)" == "<h1>Not Found</h1>" ] && echo $msg INFO fossil running OK at $url  || echo $msg WARN FOSSIL NOT RUNNING AT $url
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


