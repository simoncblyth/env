# === func-gen- : tools/carthage fgp tools/carthage.bash fgn carthage fgh tools
carthage-src(){      echo tools/carthage.bash ; }
carthage-source(){   echo ${BASH_SOURCE:-$(env-home)/$(carthage-src)} ; }
carthage-vi(){       vi $(carthage-source) ; }
carthage-env(){      elocal- ; }
carthage-usage(){ cat << EOU

Carthage
==========

::

    epsilon:~ blyth$ brew info carthage
    carthage: stable 0.29.0 (bottled), HEAD
    Decentralized dependency manager for Cocoa
    https://github.com/Carthage/Carthage
    Not installed
    From: https://github.com/Homebrew/homebrew-core/blob/master/Formula/carthage.rb
    ==> Requirements
    Build: xcode ✔
    ==> Options
    --HEAD
        Install HEAD version
    epsilon:~ blyth$ 


ruby colon means symbol
-------------------------

* https://github.com/Homebrew/homebrew-core/blob/master/Formula/carthage.rb
* https://stackoverflow.com/questions/6337897/what-is-the-colon-operator-in-ruby




EOU
}
carthage-dir(){ echo $(local-base)/env/tools/tools-carthage ; }
carthage-cd(){  cd $(carthage-dir); }
carthage-mate(){ mate $(carthage-dir) ; }
carthage-get(){
   local dir=$(dirname $(carthage-dir)) &&  mkdir -p $dir && cd $dir

}
