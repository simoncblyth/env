# === func-gen- : offline/gendbi fgp offline/gendbi.bash fgn gendbi fgh offline
gendbi-src(){      echo offline/gendbi/gendbi.bash ; }
gendbi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gendbi-src)} ; }
gendbi-vi(){       vi $(gendbi-source) ; }
gendbi-srcdir(){   echo $(dirname $(gendbi-source)) ; }
gendbi-env(){      elocal- ; }
gendbi-usage(){
  cat << EOU
     gendbi-src : $(gendbi-src)
     gendbi-dir : $(gendbi-dir)


EOU
}
gendbi-dir(){ echo $(local-base)/env/offline/offline-gendbi ; }
gendbi-cd(){  cd $(gendbi-dir); }
gendbi-mate(){ mate $(gendbi-dir) ; }

gendbi-spec-(){ echo ${GENDBI_SPEC:-SimPmtSpec} ; }
gendbi-spec(){  cat $(gendbi-srcdir)/spec/$(gendbi-spec-).spec ; }
gendbi-tmpl(){  echo SubDbiTableRow ; }

gendbi-parse(){      gendbi-spec | python $(gendbi-srcdir)/parse.py  ; }
gendbi-gen(){        gendbi-spec | python $(gendbi-srcdir)/gendbi.py $(gendbi-tmpl).$1 ; }

gendbi-h(){          gendbi-gen h  ; } 
gendbi-cc(){         gendbi-gen cc ; }
gendbi-sql(){        gendbi-gen sql ; }
gendbi-tex(){        gendbi-gen tex ; }
gendbi-tracwiki(){   gendbi-gen tracwiki ; }
gendbi-mediawiki(){  gendbi-gen mediawiki ; }


gendbi-pdf(){
  local iwd=$PWD
  local tmp=/tmp/$USER/env/$FUNCNAME/$(gendbi-spec-).tex && mkdir -p $(dirname $tmp)
  cd $(dirname $tmp)
  gendbi-tex > $tmp
  pdflatex $tmp
}






