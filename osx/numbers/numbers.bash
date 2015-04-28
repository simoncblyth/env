# === func-gen- : osx/numbers/numbers fgp osx/numbers/numbers.bash fgn numbers fgh osx/numbers
numbers-src(){      echo osx/numbers/numbers.bash ; }
numbers-source(){   echo ${BASH_SOURCE:-$(env-home)/$(numbers-src)} ; }
numbers-vi(){       vi $(numbers-source) ; }
numbers-env(){      elocal- ; }
numbers-usage(){ cat << EOU

NUMBERS
========

Although applescripts are extremely slow compared to using python approaches with 
xlwt/xlrd/xlutils the Applescript advantage is that formatting of the spreadsheet is 
retained (to the extent that Numbers.app is able to retain it).

Essentially the applescript duplicates the actions that could be done manually 
so the structure of the resultant updated spreadsheet should match those created by 
laborious manual cell-by-cell data entry. This is not the case with python
tools : *xlutils.copy.copy* creates a fresh copy and loosing formatting information
and potentially other things.


Usage Steps
------------

#. Open spreadsheet with Numbers.app
#. Check that the frontmost spreadsheet contains a worksheet named *numbers-sheetname* 
#. To export a CSV file into PWD::

     numbers-export out.csv

#. To import into the current spreadsheet::

     numbers-import in.csv

Example::

    delta:Desktop blyth$ open Example_eng.xls 
    delta:Desktop blyth$ numbers-
    delta:Desktop blyth$ numbers-import export.csv 
    === numbers-import : path /Users/blyth/Desktop/export.csv
    sheetname Journal paper toprow 3

Issues
-------

#. "volume and page number" fields giving "0" when look 
   like mathematical expression eg "99/10000". 
   This only happens when importing into the empty template spreadsheet, 
   not when adding rows to a spreadsheet existing entries.

**NB**
-------

The import/export CSV use character "|" as delimiter so fields 
cannot include that character.  There is currently no checking of that. 

TODO
-----

#. check for presense of the delimiter character in fields and refuse to export 

RANT
----
 
Why require input in proprietry undocumented binary file formats ? Why not something simple like CSV ? 

 
FUNCTIONS
----------

*numbers-import path*
       import CSV path into the frontmost Numbers.app spreadsheet  
       
*numbers-export path*
       export CSV path from the frontmost Numbers.app spreadsheet, exporting
       is much much faster than importing
 
*numbers-context* 
       dump the sheetname and toprow that is imported/exported



EOU
}
numbers-dir(){ echo $(local-base)/env/osx/numbers ; }
numbers-sdir(){ echo $(env-home)/osx/numbers ; }
numbers-cd(){  cd $(numbers-sdir); }
numbers-scd(){  cd $(numbers-sdir); }
numbers-mate(){ mate $(numbers-dir) ; }

numbers-sheetname(){ echo ${NUMBERS_SHEETNAME:-"Sheet 1"} ; }
numbers-toprow(){    echo ${NUMBERS_TOPROW:-3} ; }
numbers-delimiter(){ echo ${NUMBERS_DELIMITER:-TAB} ; }   # TAB is special cased in the scripts
numbers-context(){   cat << EOC
sheetname $(numbers-sheetname) toprow $(numbers-toprow)
EOC
}

numbers-import(){
   local msg="=== $FUNCNAME :"
   local path=${1:-import.csv}
   path=$(realpath $path)    # must exist for realpath to work
   echo $msg path $path
   numbers-context
   osascript $(numbers-sdir)/numbers_import_csv.applescript $path "$(numbers-sheetname)" $(numbers-toprow) $(numbers-delimiter)
}

numbers-absolute-path(){
   case $1 in 
     /*) echo $1 ;;
      *) echo $PWD/$1 ;; 
   esac
}

numbers-export(){
   local msg="=== $FUNCNAME :"
   local path=${1:-export.csv}
   path=$(numbers-absolute-path $path)
   echo $msg path $path
   numbers-context
   osascript $(numbers-sdir)/numbers_export_csv.applescript $path "$(numbers-sheetname)" $(numbers-toprow) $(numbers-delimiter)
}

numbers-document-name(){
   osascript $(numbers-sdir)/numbers_document_name.applescript 
}

numbers-to-csv(){
   local msg="=== $FUNCNAME :"
   local dir=${1:-$PWD}
   local name=$(numbers-document-name)
   local path="$dir/$name/$name.csv"

   local odir=$(dirname $path)
   if [ ! -d "$odir" ]; then 
      mkdir -p $odir
   fi 

   echo $msg write frontmost Numbers document $name into "$path" assuming sheetname $(numbers-sheetname) toprow $(numbers-toprow) delimiter $(numbers-delimiter) 
   [ -f "$path" ] && echo $msg there is preexisting "$path" : remove it and rerun && return  
   numbers-export "$path"
}


