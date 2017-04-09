# === func-gen- : base/vim/vim fgp base/vim/vim.bash fgn vim fgh base/vim
vim-src(){      echo base/vim/vim.bash ; }
vim-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vim-src)} ; }
vim-vi(){       vi $(vim-source) ; }
vim-env(){      elocal- ; }
vim-usage(){ cat << EOU

VIM Tips
=========


* http://www.astrohandbook.com/ch20/vi_guide.html


Spell Checking
----------------

* http://thejakeharding.com/tutorial/2012/06/13/using-spell-check-in-vim.html

Add below to ~/.vimrc::

    set spelllang=en
    set spellfile=~/.vim/en.utf-8.add

Enable spell checking with::

   :set spell

Navigate:

   ]s   ## move to next "mispelled" word
   zg   ## add current selected mispelling to ok list 



Windows MSYS2 Arrow keys
-------------------------

Arrow keys introduce funny chars in insert mode unless you use::

   :set term=builtin_ansi


Tabulating text with aligned columns
--------------------------------------

Vim ways to do this either need plugins or look too involved, so do on command line with "column"
::

    delta:~ blyth$ pbpaste | cat
        DBNS_HALL5_TEMP DBNS_H5_Temp_PT1            Low!!!             -1.00   2016-05-06 20:05:26
        DBNS_HALL5_TEMP        DBNS_H5_Temp_PT4            Low!!!             -1.00   2016-05-06 20:05:26
        DBNS_HALL5_TEMP        DBNS_H5_Temp_PT2            Low!!!             -1.00   2016-05-06 20:05:26
        DBNS_HALL5_TEMP        DBNS_H5_Temp_PT3            Low!!!             -1.00   2016-05-06 20:05:26 

    delta:~ blyth$ pbpaste | column -t 
    DBNS_HALL5_TEMP  DBNS_H5_Temp_PT1  Low!!!  -1.00  2016-05-06  20:05:26
    DBNS_HALL5_TEMP  DBNS_H5_Temp_PT4  Low!!!  -1.00  2016-05-06  20:05:26
    DBNS_HALL5_TEMP  DBNS_H5_Temp_PT2  Low!!!  -1.00  2016-05-06  20:05:26
    DBNS_HALL5_TEMP  DBNS_H5_Temp_PT3  Low!!!  -1.00  2016-05-06  20:05:26



Gutter Line Numbers
----------------------

* http://vim.wikia.com/wiki/Display_line_numbers


Perl Tricks
-------------

String replacement in all files containing the string::

    simon:opticks blyth$ opticks-lfind NGLMStream 
    ./opticksnpy/tests/NBBoxTest.cc
    ./opticksnpy/tests/NBoxTest.cc
    ./opticksnpy/tests/NFieldGrid3Test.cc
    ./opticksnpy/tests/NGLMStreamTest.cc
    ./opticksnpy/tests/NGLMTest.cc
    ./opticksnpy/tests/NNodeTest.cc
    ./opticksnpy/tests/NSphereTest.cc
    ./opticksnpy/NBox.cpp
    ./opticksnpy/NCSG.cpp
    ./opticksnpy/NFieldGrid3.cpp
    ./opticksnpy/NGLMStream.cpp
    ./opticksnpy/NGrid3.cpp
    ./opticksnpy/NOctools.cpp
    ./opticksnpy/NSphere.cpp
    ./opticksnpy/CMakeLists.txt
    ./opticksnpy/tests/CMakeLists.txt

    simon:opticks blyth$ perl -pi -e 's,NGLMStream,NGLMExt,g' `!!`
    perl -pi -e 's,NGLMStream,NGLMExt,g' `opticks-lfind NGLMStream `



Regexp Replace
---------------

enum into switch
~~~~~~~~~~~~~~~~~~

For example turning an enum into a switch statement::

    .,+20s/\s*\(\S*\).*/case \1 : s="\1" ;break; /gc

enum into string consts
~~~~~~~~~~~~~~~~~~~~~~~~~

Replace enum codes starting ERROR in the next 10 lines with string consts::

    .,+10s/\s*\(ERROR\w*\).*$/static const char* \1_ = "\1" ;/gc

For example replacing::

     enum {
         ERROR_LHS_POP_EMPTY         = 0x1 << 0,  
         ERROR_RHS_POP_EMPTY         = 0x1 << 1,  
         ERROR_LHS_END_NONEMPTY      = 0x1 << 2,  
         ERROR_RHS_END_EMPTY         = 0x1 << 3,
         ERROR_BAD_CTRL              = 0x1 << 4,
         ERROR_LHS_OVERFLOW          = 0x1 << 5,
         ERROR_RHS_OVERFLOW          = 0x1 << 6,
         ERROR_LHS_TRANCHE_OVERFLOW  = 0x1 << 7,
         ERROR_RHS_TRANCHE_OVERFLOW  = 0x1 << 8
     }

With::

    enum {
    static const char* ERROR_LHS_POP_EMPTY_ = "ERROR_LHS_POP_EMPTY" ;
    static const char* ERROR_RHS_POP_EMPTY_ = "ERROR_RHS_POP_EMPTY" ;
    static const char* ERROR_LHS_END_NONEMPTY_ = "ERROR_LHS_END_NONEMPTY" ;
    static const char* ERROR_RHS_END_EMPTY_ = "ERROR_RHS_END_EMPTY" ;
    static const char* ERROR_BAD_CTRL_ = "ERROR_BAD_CTRL" ;
    static const char* ERROR_LHS_OVERFLOW_ = "ERROR_LHS_OVERFLOW" ;
    static const char* ERROR_RHS_OVERFLOW_ = "ERROR_RHS_OVERFLOW" ;
    static const char* ERROR_LHS_TRANCHE_OVERFLOW_ = "ERROR_LHS_TRANCHE_OVERFLOW" ;
    static const char* ERROR_RHS_TRANCHE_OVERFLOW_ = "ERROR_RHS_TRANCHE_OVERFLOW" ;
    }


enum into stringstream dump
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly replace enum with stringstream-ing the consts::

    .,+10s/\s*\(ERROR\w*\).*$/if(err \& \1 ) ss << \1_ << " " ;/gc

    enum { 
    if(err & ERROR_LHS_POP_EMPTY ) ss << ERROR_LHS_POP_EMPTY_ << " " ;
    if(err & ERROR_RHS_POP_EMPTY ) ss << ERROR_RHS_POP_EMPTY_ << " " ;
    if(err & ERROR_LHS_END_NONEMPTY ) ss << ERROR_LHS_END_NONEMPTY_ << " " ;
    if(err & ERROR_RHS_END_EMPTY ) ss << ERROR_RHS_END_EMPTY_ << " " ;
    if(err & ERROR_BAD_CTRL ) ss << ERROR_BAD_CTRL_ << " " ;
    if(err & ERROR_LHS_OVERFLOW ) ss << ERROR_LHS_OVERFLOW_ << " " ;
    if(err & ERROR_RHS_OVERFLOW ) ss << ERROR_RHS_OVERFLOW_ << " " ;
    if(err & ERROR_LHS_TRANCHE_OVERFLOW ) ss << ERROR_LHS_TRANCHE_OVERFLOW_ << " " ;
    if(err & ERROR_RHS_TRANCHE_OVERFLOW ) ss << ERROR_RHS_TRANCHE_OVERFLOW_ << " " ;
    }





Switch text to lower/upper case
--------------------------------

Visually select, then  

* U to convert to uppercase 
* u to convert to lowecase 



Overwrite Mode : handy for ascii art
--------------------------------------

* shift-R



EOU
}
vim-dir(){ echo $(local-base)/env/base/vim/base/vim-vim ; }
vim-cd(){  cd $(vim-dir); }


vim-vimrc(){ cat << EOR
syntax on

set nu
set paste

set smartindent
set tabstop=4
set shiftwidth=4
set expandtab

set term=builtin_ansi

EOR
}




