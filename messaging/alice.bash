# === func-gen- : messaging/alice fgp messaging/alice.bash fgn alice fgh messaging
alice-src(){      echo messaging/alice.bash ; }
alice-source(){   echo ${BASH_SOURCE:-$(env-home)/$(alice-src)} ; }
alice-vi(){       vi $(alice-source) ; }
alice-env(){      elocal- ; }
alice-usage(){
  cat << EOU
     alice-src : $(alice-src)
     alice-dir : $(alice-dir)


EOU
}
alice-dir(){ echo $(local-base)/env/messaging/alice ; }
alice-cd(){  cd $(alice-dir); }
alice-mate(){ mate $(alice-dir) ; }
alice-get(){
   local dir=$(dirname $(alice-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://github.com/auser/alice.git
}

alice-kludge-for-ancient-erlang(){
   alice-cd
   find . -name '*.erl' -exec perl -pi -e 's,string:to_lower,httpd_util:to_lower,g' {} \;
}

alice-build(){
   alice-get
   alice-cd
   alice-kludge-for-ancient-erlang
   ./start.sh
}


alice-home(){ echo /var/lib/rabbitmq ; }
alice-run(){ sudo -u rabbitmq env HOME=$(alice-home) erl -pa $(alice-dir)/ebin -pa $(alice-dir)/deps/*/ebin -sname alice -s reloader -boot alice -setcookie $(sudo cat $(alice-home)/.erlang.cookie) ; }
alice-test(){ curl -i http://localhost:9999/vhosts ; }
