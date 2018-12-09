# === func-gen- : tools/cocoapods fgp tools/cocoapods.bash fgn cocoapods fgh tools
cocoapods-src(){      echo tools/cocoapods.bash ; }
cocoapods-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cocoapods-src)} ; }
cocoapods-vi(){       vi $(cocoapods-source) ; }
cocoapods-env(){      elocal- ; }
cocoapods-usage(){ cat << EOU

Cocoapods
============

* https://cocoapods.org/
* https://www.raywenderlich.com/156971/cocoapods-tutorial-swift-getting-started
* ~/tree/gems/cocoapods.log


1.5G of very many json files (for every version of every pod) in HOME/.cocoapods::

    epsilon:~ blyth$ du -hs .cocoapods
    1.5G	.cocoapods

::

    epsilon:~ blyth$ which pod
    /usr/local/bin/pod

::

    epsilon:~ blyth$ ll .cocoapods/repos/master/
    total 24
    drwxr-xr-x   3 blyth  staff   96 Apr 10 21:44 ..
    -rw-r--r--   1 blyth  staff   38 Apr 10 21:50 .gitignore
    -rw-r--r--   1 blyth  staff   55 Apr 10 21:50 CocoaPods-version.yml
    -rw-r--r--   1 blyth  staff  575 Apr 10 21:50 README.md
    drwxr-xr-x   7 blyth  staff  224 Apr 10 21:50 .
    drwxr-xr-x  18 blyth  staff  576 Apr 10 21:51 Specs
    drwxr-xr-x  13 blyth  staff  416 Apr 10 21:51 .git
    epsilon:~ blyth$ 

::

    epsilon:~ blyth$ l /usr/local/bin/
    total 48
    -rwxr-xr-x  1 root  wheel  -  534 Apr 10 21:40 sandbox-pod
    -rwxr-xr-x  1 root  wheel  -  526 Apr 10 21:40 pod
    -rwxr-xr-x  1 root  wheel  -  532 Apr 10 21:40 xcodeproj
    -rwxr-xr-x  1 root  wheel  -  538 Apr 10 21:40 fuzzy_match
    -r-xr-xr-x  1 root  wheel  - 7686 Dec 20 19:54 uninstall_cuda_drv.pl
    epsilon:~ blyth$ 





EOU
}
cocoapods-dir(){ echo $(local-base)/env/tools/tools-cocoapods ; }
cocoapods-cd(){  cd $(cocoapods-dir); }
cocoapods-mate(){ mate $(cocoapods-dir) ; }
cocoapods-get(){
   local dir=$(dirname $(cocoapods-dir)) &&  mkdir -p $dir && cd $dir

}
