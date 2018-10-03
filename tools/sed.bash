# === func-gen- : tools/sed fgp tools/sed.bash fgn sed fgh tools src base/func.bash
sed-source(){   echo ${BASH_SOURCE} ; }
sed-edir(){ echo $(dirname $(sed-source)) ; }
sed-ecd(){  cd $(sed-edir); }
sed-dir(){  echo $LOCAL_BASE/env/tools/sed ; }
sed-cd(){   cd $(sed-dir); }
sed-vi(){   vi $(sed-source) ; }
sed-env(){  elocal- ; }
sed-usage(){ cat << EOU

sed
=====

* http://www.catonmat.net/blog/wp-content/uploads/2008/09/sed1line.txt

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



EOU
}
sed-get(){
   local dir=$(dirname $(sed-dir)) &&  mkdir -p $dir && cd $dir

}
