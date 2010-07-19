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

gendbi-spec-(){ echo SimPmtSpec ; }
gendbi-spec(){  cat $(gendbi-srcdir)/spec/$(gendbi-spec-).spec ; }
gendbi-tmpl(){  echo SubDbiTableRow ; }

gendbi-parse(){      gendbi-spec | python $(gendbi-srcdir)/parse.py  ; }
gendbi-h(){          gendbi-spec | python $(gendbi-srcdir)/gendbi.py $(gendbi-tmpl).h   ; }
gendbi-cc(){         gendbi-spec | python $(gendbi-srcdir)/gendbi.py $(gendbi-tmpl).cc  ; }
gendbi-sql(){        gendbi-spec | python $(gendbi-srcdir)/gendbi.py $(gendbi-tmpl).sql ; }
gendbi-tex(){        gendbi-spec | python $(gendbi-srcdir)/gendbi.py $(gendbi-tmpl).tex ; }
gendbi-tracwiki(){   gendbi-spec | python $(gendbi-srcdir)/gendbi.py $(gendbi-tmpl).tracwiki ; }
gendbi-mediawiki(){  gendbi-spec | python $(gendbi-srcdir)/gendbi.py $(gendbi-tmpl).mediawiki ; }

gendbi-pdf(){
  local iwd=$PWD
  local tmp=/tmp/$USER/env/$FUNCNAME/$(gendbi-spec-).tex && mkdir -p $(dirname $tmp)
  cd $(dirname $tmp)
  gendbi-tex > $tmp
  pdflatex $tmp
}






