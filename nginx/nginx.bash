# === func-gen- : nginx/nginx.bash fgp nginx/nginx.bash fgn nginx
nginx-src(){      echo nginx/nginx.bash ; }
nginx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nginx-src)} ; }
nginx-vi(){       vi $(nginx-source) ; }
nginx-env(){      elocal- ; env-append $(nginx-sbin) ; }



nginx-usage(){
  cat << EOU
     nginx-src : $(nginx-src)

     http://wiki.nginx.org/Main



EOU
}

nginx-name(){ echo nginx-0.7.61 ; }
nginx-url(){  echo http://sysoev.ru/nginx/$(nginx-name).tar.gz ; }
nginx-dir(){  echo $(local-base)/env/nginx/$(nginx-name) ; }
nginx-get(){
   local dir=$(dirname $(nginx-dir)) && mkdir -p $dir
   cd $dir
   [ ! -d "$(nginx-name)" ] && curl -L -O $(nginx-url) && tar zxvf $(nginx-name).tar.gz  
}

nginx-prefix(){
  case ${1:-$NODE_TAG} in 
     C) echo /usr/local/nginx ;;
  esac
}
nginx-sbin(){  echo $(nginx-prefix $*)/sbin ; }
nginx-rconf(){ echo conf/nginx.conf ; }
nginx-conf(){  echo $(nginx-prefix $*)/$(nginx-rconf) ; }
nginx-edit(){  sudo vim $(nginx-conf) ; }

nginx-ps(){ ps aux | grep nginx ; }

nginx-build(){
   cd $(nginx-dir)

   ## using default prefix of /usr/local/nginx
   ./configure
   make
   sudo make install
}

nginx-rproxy-(){  cat << EOC

worker_processes 1 ;

events { worker_connections  1024; }

http {

    server {
        listen       80;
        server_name localhost ;
        location / {
              proxy_pass     http://picasaweb.google.com ;
              proxy_redirect http://picasaweb.google.com/ /;
              proxy_set_header  X-Real-IP  \$remote_addr;
        }
    }
}

EOC

}



nginx-rproxy-rconf(){  echo conf/local/rproxy.conf ; }
nginx-rproxy-conf(){   echo $(nginx-prefix)/$(nginx-rproxy-rconf) ; }
nginx-rproxy(){
   local msg="=== $FUNCNAME :"
   local conf=$(nginx-rproxy-conf)  
   local tmp=/tmp/env/$FUNCNAME/$(basename $conf) && mkdir -p $(dirname $tmp)
   $FUNCNAME- > $tmp

   echo $msg wrote $tmp ... replace $conf ?
   cat $tmp

   local cmd="sudo mkdir -p $(dirname $conf) && sudo cp $tmp $conf "
   echo $msg $cmd
   eval $cmd

}

nginx-rproxy-start(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo nginx -c $(nginx-rproxy-rconf) "
   echo $cmd
   eval $cmd 
    
}


nginx-stop(){
    sudo nginx -s stop
}


