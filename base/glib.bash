# === func-gen- : base/glib fgp base/glib.bash fgn glib fgh base
glib-src(){      echo base/glib.bash ; }
glib-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glib-src)} ; }
glib-vi(){       vi $(glib-source) ; }
glib-env(){      elocal- ; }
glib-usage(){
  cat << EOU
     glib-src : $(glib-src)
     glib-dir : $(glib-dir)

     http://library.gnome.org/devel/glib/stable/


EOU
}
glib-dir(){ echo $(local-base)/env/base/glib ; }
glib-cd(){  cd $(glib-dir); }
glib-mate(){ mate $(glib-dir) ; }
glib-get(){
   local dir=$(dirname $(glib-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://git.gnome.org/glib

}
