# === func-gen- : cpprun/threading/threading fgp cpprun/threading/threading.bash fgn threading fgh cpprun/threading src base/func.bash
threading-source(){   echo ${BASH_SOURCE} ; }
threading-edir(){ echo $(dirname $(threading-source)) ; }
threading-ecd(){  cd $(threading-edir); }
threading-dir(){  echo $LOCAL_BASE/env/cpprun/threading/threading ; }
threading-cd(){   cd $(threading-dir); }
threading-vi(){   vi $(threading-source) ; }
threading-env(){  elocal- ; }
threading-usage(){ cat << EOU


* :google:`std::thread boost::thread pthread`

* https://stackoverflow.com/questions/63705018/boostthread-vs-stdthread-vs-pthread

Note that std::thread itself doesn't need to be used directly. The standard has
useful abstractions such as std::reduce, std::packaged_task, std::async,
parallel execution policies for algorithms etc.



* https://computing.llnl.gov/tutorials/pthreads/#Abstract


EOU
}
threading-get(){
   local dir=$(dirname $(threading-dir)) &&  mkdir -p $dir && cd $dir

}
