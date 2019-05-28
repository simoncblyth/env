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



Get rid of tabs : eg from diff
---------------------------------

::

   :retab








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


Key remapping now that w key is playing up, to move between splits
--------------------------------------------------------------------

* :google:`vim move between splits`
* https://stackoverflow.com/questions/3776117/what-is-the-difference-between-the-remap-noremap-nnoremap-and-vnoremap-mapping

nnoremap
   is a normal mode non-recursive mapping 

Add to .vimrc::

    " https://robots.thoughtbot.com/vim-splits-move-faster-and-more-naturally
    " https://github.com/thoughtbot/dotfiles/blob/master/vimrc
    " navigate splits with ctrl-j/k/l/h
    nnoremap <C-J> <C-W><C-J>
    nnoremap <C-K> <C-W><C-K>
    nnoremap <C-L> <C-W><C-L>
    nnoremap <C-H> <C-W><C-H>





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




How to paste a column in blockwise (rather than tedious linewise fashion) using VISUAL BLOCK 
---------------------------------------------------------------------------------------------

* https://stackoverflow.com/questions/9120552/how-do-i-paste-a-column-of-text-after-a-different-column-of-text-in-vim


TIPS:

* make sure there are spaces on the last line of the block to cover the 
  entire maximum width of what you want to cut/paste

* VISUAL-BLOCK (ctrl-v) is not the same as the usual VISUAL (v) mode


::

    Names                
    Donald Knuth
    Sebastian Thrun
    Peter Norvig
    Satoshi Nakamoto

    Age
    100
    50
    60
    45


    Names                    Age 
    Donald Knuth             100
    Sebastian Thrun          50
    Peter Norvig             60
    Satoshi Nakamoto         45



1. Yank it in visual mode:

   * Move cursor to the beginning of Age
   * Press Ctrl + v to enter *VISUAL BLOCK* mode 
     (a rectangular block should highlight, that you adjust via cursor positioning)  
   * Move cursor to 5 in 45
   * Press y to yank (or d to delete), you have now yanked in visual mode.

2. Paste (in normal mode)

   * Move to the end of the first line and add more spaces because it's shorter than the second line for example. 
     If you paste a "block" without adding extra spaces, it will overwrite the "run" in Sebastian Thrun.

   * Now you're on the first line, insert a few spaces after the last character. 
     Make sure you're not in insert mode and hit p to paste the block. (If you want to paste in insert mode, use ctrl+r ")





Regexp Replace
---------------


vim substitute tips
~~~~~~~~~~~~~~~~~~~~~

* http://vim.wikia.com/wiki/Search_and_replace


add a parameter to a method call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note trying to escape the brackets prevents this from working::

    :.,+20s/));/),epsilon);/g


vim substitute this line only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

   :s/cls/self/g


switch case into else if 
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   :.,+20s/\s*case \(\S*\):\s*$/else if(node->type == \1)/gc 


enum into switch
~~~~~~~~~~~~~~~~~~

For example turning an enum into a switch statement::

    .,+20s/\s*\(\S*\).*/case \1 : s="\1" ;break; /gc

enum into string consts
~~~~~~~~~~~~~~~~~~~~~~~~~

Replace enum codes starting ERROR in the next 10 lines with string consts::

    .,+10s/\s*\(FIND\w*\).*$/static const char* \1_ = "\1" ;/gc
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




Copying Lines Around
-----------------------

* https://stackoverflow.com/questions/4533530/vim-replacing-a-line-with-another-one-yanked-before

::

      1 #ifdef OLD_PARAMETERS
      2 BParameters* getParam();
      3 #else
      4 NMeta* getParam();
      5 #endif
      6 
      7 
      8 
      9 // <---- the line after which to insert in 9  
     10 BParameters* getParamA();
     11 
     12 
     13 
     14 
     15 BParameters* getParamB();
     16 
     17 
     18 
     19 
        

Then 1t9 copying line 1 to be inserted after line 9 (and before line 10)::

      1 #ifdef OLD_PARAMETERS
      2 BParameters* getParam();
      3 #else
      4 NMeta* getParam();
      5 #endif
      6 
      7 
      8 
      9 
     10 #ifdef OLD_PARAMETERS
     11 BParameters* getParamA();
     12 
     13 
     14 
     15 // <--- the line after which to insert is 15 
     16 BParameters* getParamB();
     17 
     18 
     19 
     20 
     21 
     22 
      
Then 1t15 copying line 1 to the line after 15 in the above::

      1 #ifdef OLD_PARAMETERS
      2 BParameters* getParam();
      3 #else
      4 NMeta* getParam();
      5 #endif
      6 
      7 
      8 
      9 
     10 #ifdef OLD_PARAMETERS
     11 BParameters* getParamA();
     12 
     13 
     14 
     15 
     16 #ifdef OLD_PARAMETERS
     17 BParameters* getParamB();
     18 
     19 
     20 
     21 
     22 
     23 

Now copy an inclusive range of lines :3,5t11 to the line after 11 and before 12::

     01 #ifdef OLD_PARAMETERS
      2 BParameters* getParam();
      3 #else
      4 NMeta* getParam();
      5 #endif
      6 
      7 
      8 
      9 
     10 #ifdef OLD_PARAMETERS
     11 BParameters* getParamA();
     12 #else
     13 NMeta* getParam();
     14 #endif
     15 
     16 
     17 
     18 
     19 #ifdef OLD_PARAMETERS
     20 BParameters* getParamB();
     21 
     22 
     23 
     24 
     25 












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




