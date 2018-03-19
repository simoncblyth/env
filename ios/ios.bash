# === func-gen- : ios/ios fgp ios/ios.bash fgn ios fgh ios
ios-src(){      echo ios/ios.bash ; }
ios-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ios-src)} ; }
ios-vi(){       vi $(ios-source) ; }
ios-env(){      elocal- ; }
ios-usage(){ cat << EOU

iOS
====

* http://arstechnica.com/apple/2016/01/new-chrome-for-ios-is-finally-as-fast-and-stable-as-safari/


* iOS 8 WKWebView exposes faster Nitro rendering than UIWebView

* https://iosdevdirectory.com/



EOU
}
ios-dir(){ echo $(local-base)/env/ios/ios-ios ; }
ios-cd(){  cd $(ios-dir); }
ios-mate(){ mate $(ios-dir) ; }
ios-get(){
   local dir=$(dirname $(ios-dir)) &&  mkdir -p $dir && cd $dir

}
