# === func-gen- : network/turbovnc/turbovnc fgp network/turbovnc/turbovnc.bash fgn turbovnc fgh network/turbovnc
turbovnc-src(){      echo network/turbovnc/turbovnc.bash ; }
turbovnc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(turbovnc-src)} ; }
turbovnc-vi(){       vi $(turbovnc-source) ; }
turbovnc-env(){      elocal- ; }
turbovnc-usage(){ cat << EOU


* http://turbovnc.org

* http://downloads.sourceforge.net/project/turbovnc/2.0/TurboVNC-2.0-OracleJava.dmg

Web download yields::

   /Users/blyth/Downloads/TurboVNC-2.0-OracleJava.dmg

Opening::

    open /Users/blyth/Downloads/TurboVNC-2.0-OracleJava.dmg

Record pkg contents::

    turbovnc-pkgpath-lsbom > ~/packages/turbovnc-pkgpath-lsbom.txt



EOU
}
turbovnc-dir(){ echo $(local-base)/env/network/turbovnc/network/turbovnc-turbovnc ; }
turbovnc-cd(){  cd $(turbovnc-dir); }
turbovnc-mate(){ mate $(turbovnc-dir) ; }
turbovnc-get(){
   local dir=$(dirname $(turbovnc-dir)) &&  mkdir -p $dir && cd $dir

}

turbovnc-pkgpath(){
   echo /Volumes/TurboVNC-2.0/TurboVNC.pkg 
}
turbovnc-pkgpath-lsbom(){
   lsbom "$(pkgutil --bom "$(turbovnc-pkgpath)")"
}
turbovnc-html(){
   open /Library/Documentation/TurboVNC/index.html
}
turbovnc-viewer(){
   open /Applications/TurboVNC/TurboVNC\ Viewer.app
}
