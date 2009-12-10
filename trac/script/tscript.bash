
tscript-env(){    echo -n ; }
tscript-src(){    echo trac/script/tscript.bash ; }
tscript-source(){ echo ${BASH_SOURCE:-$(env-home)/$(tscript-src)} ; }  
tscript-vi(){     vi $(tscript-source) ; }
tscript-usage(){
  cat << EOU

    tscript-t2l path/to/wikitext/file
        convert to latex, putting the 

    echo "== Hello ==" | tscript-t2l

   this needs to run under the same python that Trac is using otherwise
   will get no such module "trac2latex"

EOU
}

tscript-t2l-py(){ echo $(env-home)/trac/script/t2l.py ; }
tscript-t2l(){
   case "$#" in
      0) cat - | python `tscript-t2l-py` ;;
      *) python `tscript-t2l-py` $* ;;
   esac  
}

tscript-t2l-test(){
  tscript-t2l /Users/blyth/workflow/admin/reports/nuu-report-oct-2008.txt
}

