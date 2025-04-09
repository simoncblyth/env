# === func-gen- : tools/sed fgp tools/sed.bash fgn sed fgh tools src base/func.bash
sed-source(){   echo ${BASH_SOURCE} ; }
sed-edir(){ echo $(dirname $(sed-source)) ; }
sed-ecd(){  cd $(sed-edir); }
sed-dir(){  echo $LOCAL_BASE/env/tools/sed ; }
sed-cd(){   cd $(sed-dir); }
sed-vi(){   vi $(sed-source) + $(sed-edir)/sed1line.txt ; }
sed-env(){  elocal- ; }
sed-usage(){ cat << EOU

sed
=====

* http://www.catonmat.net/blog/wp-content/uploads/2008/09/sed1line.txt



when sed too difficult use python
------------------------------------

::


   #sed -E 's/^(sSurftube|svacSurftube|solidSJ|solidXJ).*/#&/' $meshname > $tessname 
   # HMM: cannot negate a match with sed, so do it line-by-line with python

   oneline="import sys,re;ptn=re.compile('^(sSurftube|svacSurftube|solidSJ|solidXJ).*');print('\n'.join([('%s' if ptn.match(line) else '#%s') % line.rstrip() for line in sys.stdin.readlines()]))" 
   cat $meshname | python -c "$oneline" - > $tessname



remove comments from here doc lines 
--------------------------------------

::

    vip-paths-(){ sed 's/#.*//' << EOP

       $HOME/.bash_profile            # top level umbrella : DONT PLACE DETAILS HERE 

          $G/vip/vip.bash             # bash functions needed everywhere eg for prompt and ssh environment
          $G/.local.bash              # machine specifics eg setting up for compiler

    EOP
    }


Test file
-----------

::

    cat demo.txt 
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10


Reverse lines 
----------------

::

   cat demo.txt | sed '1!G;h;$!d'


Quote All Lines of a File : eg to handle filesystem paths with spaces in them when passing to xargs
----------------------------------------------------------------------------------------------------

::

    sed -e 's/^/"/' -e 's/$/"/' test.txt



Extract a range of lines from a file
-------------------------------------

    


Extract a stretch of lines containing paths and quote them : eg when contain spaces
--------------------------------------------------------------------------------------

::

    sed -n -e 's/^/"/' -e 's/$/"/' -e 1000,1010p  test.txt 
    sed -n -e 's/^/"/' -e 's/$/"/' -e 1000,1010p  test.txt | xargs open -n 


    sed -n '1011q;1000,1010p' test.txt 
    # supposed faster to quit at the line beyond selection 

    sed -n -e 's/^/"/' -e 's/$/"/'  -e "1011q;1000,1010p"  test.txt


    sed -n 's/^/"/;s/$/"/;1011q;1000,1010p'  test.txt
    # can be used a one compound commans  

    sed -n 's/^/"/;s/$/"/;1000,1010p;1011q'  test.txt
    # hmm does the order make any difference : seems not 

    sed -n "s/^/\"/;s/$/\"/;${idx0},${idx1}p;${idx2}q"  test.txt | xargs stat -l
    # needs escaping when use with substitution 



Extract a line from a file
-----------------------------

::

    sed "5q;d" demo.txt 
    5


Emit to stdout the file with a line skipped
---------------------------------------------

::

    sed "5d" demo.txt 
    1
    2
    3
    4
    6
    7
    8
    9
    10

Skip a bunch of lines::

    sed "5d;4d;8d;1d;7d" demo.txt 
    2
    3
    6
    9
    10



Inplace removal of lines from file
----------------------------------------

::

    epsilon:tt blyth$ sed -i "" "5d" demo.txt 
    epsilon:tt blyth$ cat demo.txt 
    1
    2
    3
    4
    6
    7
    8
    9
    10
    epsilon:tt blyth$ sed -i "" "5d" demo.txt 
    epsilon:tt blyth$ cat demo.txt 
    1
    2
    3
    4
    7
    8
    9
    10


Eating away at the file you end up with a zero length one::

    epsilon:tt blyth$ sed -i '' "1d" demo.txt 
    epsilon:tt blyth$ cat demo.txt 
    epsilon:tt blyth$ ll
    -rw-r--r--   1 blyth  wheel    0 Oct  3 10:22 demo.txt
    epsilon:tt blyth$ xxd demo.txt 
    epsilon:tt blyth$ 

::

    [ -f demo.txt ] && [ -s demo.txt ] && echo file exists and has non-zero length
    [ -f demo.txt ] && [ ! -s demo.txt ] && echo file exists but has zero length  



LIKE GREP::

    env | sed -n '/^OPTICKS/p'

NOT WORKING : TRY TO FORMAT OUTPUT FROM ENV::

    env | sed -n 's/^(OPTICKS.*)=(.*)$/\1  \2/p'   


EOU
}
sed-get(){
   local dir=$(dirname $(sed-dir)) &&  mkdir -p $dir && cd $dir

}
