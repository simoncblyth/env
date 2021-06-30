# === func-gen- : proj/zike fgp proj/zike.bash fgn zike fgh proj src base/func.bash
zike-source(){   echo ${BASH_SOURCE} ; }
zike-edir(){ echo $(dirname $(zike-source)) ; }
zike-ecd(){  cd $(zike-edir); }
zike-dir(){  echo $LOCAL_BASE/env/proj/zike/neT ; }
zike-cd(){   cd $(zike-dir); }
zike-vi(){   vi $(zike-source) ; }
zike-env(){  elocal- ; }
zike-usage(){ cat << EOU

https://github.com/Wang-Zike/neT



Hey there, we’re just writing to let you know that you’ve been automatically subscribed to a repository on GitHub.

   Wang-Zike/neT
   https://github.com/Wang-Zike/neT

You’ll receive notifications for all issues, pull requests, and comments that happen inside the repository. If you would like to stop watching this repository, you can manage your settings here:

   https://github.com/Wang-Zike/neT/subscription

You can unwatch this repository immediately by clicking here:

   https://github.com/Wang-Zike/neT/unsubscribe_via_email/AAD5TZJRKY4GXOW7KWB5BNLTVHKZXANCNFSM4FVZ3K7A

You were automatically subscribed because you’ve been given access to the repository.

Thanks!


EOU
}
zike-get(){
   local dir=$(dirname $(zike-dir)) &&  mkdir -p $dir && cd $dir

   git clone git@github.com:Wang-Zike/neT.git
}
