# === func-gen- : fossil/fossil fgp fossil/fossil.bash fgn fossil fgh fossil
fossil-src(){      echo fossil/fossil.bash ; }
fossil-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fossil-src)} ; }
fossil-vi(){       vi $(fossil-source) ; }
fossil-env(){      
    elocal- ; 
}
fossil-usage(){ cat << EOU
Fossil
========

Simple, high-reliability, distributed software configuration management

  * http://www.fossil-scm.org/fossil/doc/trunk/www/index.wiki

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

EOU
}


fossil-nam(){ echo fossil-src-20130216000435 ; }
fossil-dir(){ echo $(local-base)/env/fossil/$(fossil-nam) ; }
fossil-cd(){  cd $(fossil-dir); }
fossil-mate(){ mate $(fossil-dir) ; }
fossil-get(){
   local dir=$(dirname $(fossil-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(fossil-nam)
   local tgz=$nam.tar.gz
   local url=http://www.fossil-scm.org/download/$tgz
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz
}

fossil-build(){
   fossil-cd
   mkdir -p build
   cd build
   ../configure
   make
}
fossil-bin(){ echo $(fossil-dir)/build/fossil ; }
fossil-install(){ [ ! -x $(env-home)/bin/fossil ] &&  ln -s $(fossil-bin) $(env-home)/bin/fossil ; }
fossil-xinetd-(){ 

  # try new config approach that 
  # puts config values from fossil section 
  # of ini file into namespace

  cfg-
  cfg-parse ~/.env.cnf
  cfg-sect fossil   
  local binpath=$(fossil-bin)
  cat << EOC
# default: on
# description: The fossil server packs most apache+SVN+Trac functionality into a tiny single binary 
service fossil
{
    type = UNLISTED
    port = $port
    protocol = tcp
    socket_type     = stream
    wait            = no
    user            = $user
    cps             = 1000
    server          =  $binpath
    server_args     = http $repodir
}
EOC
}

fossil-xinetd(){
   local tgt=/etc/xinetd.d/fossil
   local tmp=/tmp/$FUNCNAME/fossil && mkdir -p $(dirname $tmp)
   fossil-xinetd- > $tmp
   sudo cp $tmp $tgt
   rm $tmp
   cat $tgt
}


