# === func-gen- : base/vim/vim fgp base/vim/vim.bash fgn vim fgh base/vim
vim-src(){      echo base/vim/vim.bash ; }
vim-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vim-src)} ; }
vim-vi(){       vi $(vim-source) ; }
vim-env(){      elocal- ; }
vim-usage(){ cat << "EOU"

VIM Tips
=========


* http://www.astrohandbook.com/ch20/vi_guide.html


goto byte in file
------------------

::

    UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 964-965: invalid continuation byte

::
   
    goto 964

    zeta:home blyth$ xxd -c 32 -d -s 963 -l 1 /usr/local/home/sysadmin/backup/homersync/blyth/homersync-stay-MIGRATION-EPSILON-TO-ZETA.log 
    00000963: 0a   
     


replace trailing whitespace
----------------------------

::

    :%s/\s\+$//gc


highlight trailing whitespace
-------------------------------

::

    :set hlsearch     OR   :set hls
    /\s\+$


vim debug
----------

* https://vimways.org/2018/debugging-your-vim-config/

::


    vim -u NONE te.txt
    vim -u NORC te.txt
    vim --noplugin te.txt


    vi -u ~/.vimrc_minimal te.txt
    vim --clean  te.txt




vim abbrev
------------

* https://www.redhat.com/sysadmin/vim-abbreviations

To create abbreviations in Vim, you can use the command :abbreviate (which
itself can be abbreviated as :ab) followed by the abbreviation and the text you
want to replace it with. For example, to abbreviate the word "operating system"
as "os," use the abbreviate command like this:

:ab os operating system

Abbrev work when disable everything::

   vim -u NONE yeto.txt 
   vim --clean  te.txt  



vim key bindings in bash shell
---------------------------------

Just found this out today and its amazing. I always felt like Ctrl-a was
cumbersome and not as good as being able to use vim bindings. I found out that
you can get vim bindings in bash and zsh!

zsh: bindkey -v


bash: set -o vi

this has helped me so much!


* https://unix.stackexchange.com/questions/30454/advantages-of-using-set-o-vi
* https://www.techrepublic.com/article/using-vi-key-bindings-in-bash-and-zsh/

ESC to enter command mode then

* ^:start-of-line
* $:end-of-line



replace mode
--------------

* shift-r (remember as "R") and start typing, escape returns to normal insert mode


replace trailing whitespace on blocks of lines
------------------------------------------------

::

    :.,$s/\s+$//gc
    :%s/\s+$//gc

    :.,$s/\s*$//gc


delete multiple lines
------------------------

::
 
    4dd   # delete four lines starting from current one 



replace spaces on line with newlines
----------------------------------------

This is very handy for making VERBOSE=1 make commands understandable:: 

    :s/ /^M/g        ## enter the ^M with  ctrl-V return



vertical split
----------------

* ctrl-w v OR :vsplit


vimdiff vertical or horizontal
---------------------------------

Change default in ~/.vimrc with diffopt::

     11 " set diffopt=horizontal
     12 set diffopt=vertical

Toggle between them with:

1. ctrl-w J    (NB need to press shift to get the capital J)
2. ctrl-w H or ctrl-w L 


delete a range of lines by line number
----------------------------------------

::

   1,10357d 


count words
-------------

* to count all words in buffer, press: g ctrl-g 
* to count all words in selected block, make the selection and then press: g ctrl-g 


replace a wildcarded string with spaces : eg name with a reference
--------------------------------------------------------------------

::

   :.,688s/0x\S*/         /gc 


viminfo error
--------------

On lxslc which has an AFS setup with tokens then seems to expire after 2 minutes
Get error at exit::

    E886: Can't rename viminfo file to /afs/ihep.ac.cn/users/b/blyth/.viminfo!

Can avoid by updating the token with kklog::

    L7[blyth@lxslc710 ~]$ t kklog
    kklog () 
    { 
        type $FUNCNAME && kinit blyth && aklog -d
    }

But prefer not to enter password every two minutes

Added line to ~/.vimrc::

    syntax on

    set nu
    set paste

    set smartindent
    set tabstop=4
    set shiftwidth=4
    set expandtab

    set viminfo+=n/hpcfs/juno/junogpu/blyth/.viminfo


split a long line eg from VERBOSE=1 building on spaces
-------------------------------------------------------

::

   s/ /\r/g



moving around the line shortcuts
---------------------------------

0 : start of line
$ : end of line 
^ : first non blanck character on line 
g_ : last non blank character on line 
 

trim trailing whitespace
---------------------------

::

   :%s/\s\+$//e

   #e: avoids giving error when no match 

* https://vi.stackexchange.com/questions/454/whats-the-simplest-way-to-strip-trailing-whitespace-from-all-lines-in-a-file



vimdiff high level tips
--------------------------

0. open all folds with (zR) 
1. have one of the files a temporay and push to it as you go to reduce differnces
2. when pulling in a new block from the other with "do" it doent work from the top, 
   position cursor at the bottom of the blue missing diff region 

3. feel free to add spaces to make the diffs line up more simply, use :diffupdate 


4. "dp" diffput  
5. "do" diffobtain



merging files with vimdiff : worked example
-----------------------------------------------

::

    epsilon:~ blyth$ jps
    epsilon:PMTSim blyth$  
    epsilon:PMTSim blyth$ jdiff NNVTMaskManager
    diff /Users/blyth/junotop/offline/./Simulation/DetSimV2/PMTSim/include/NNVTMaskManager.hh /Users/blyth/j/PMTSim/NNVTMaskManager.hh
    diff /Users/blyth/junotop/offline/./Simulation/DetSimV2/PMTSim/src/NNVTMaskManager.cc /Users/blyth/j/PMTSim/NNVTMaskManager.cc

    vd /Users/blyth/junotop/offline/./Simulation/DetSimV2/PMTSim/src/NNVTMaskManager.cc /Users/blyth/j/PMTSim/NNVTMaskManager.cc


1. not folding makes diffs and merges clearer as it avoids the code jumping around
2. decide which file you are going to change and focus on grabbing what is useful from the 
   other file entering "do" (diff-obtain) with cursor placed on line below the cyan missing block  
3. many diffs without overlaps between changes will be simply done with "do" from the target buffer
4. dont be shy about having duplication between macro blocks if it makes the diffs simpler
5. use ":diffupdate" to update the coloring after making edits with "do" or "dp" 



vimdiff
--------

* http://vimdoc.sourceforge.net/htmldoc/diff.html
* https://www.youtube.com/watch?v=Eb8S_KkmLS8

vd(){ vimdiff -c "windo set nofoldenable" $* ; }




::

   vimdiff lhs.txt rhs.txt
   vimdiff -o top.txt bot.txt
   vim -d lhs.txt rhs.txt
   vim -do top.txt bot.txt

   ctrl-WW  : jump between files   : need to do this to "u" undo a "dp" change into the other file

   zR       (directly not :zR)   opens all the folds  : this helps because it prevents jumping around when moving changes  

   ]c    next diff
   [c    prev diff

   :diffget        do   (diff-obtain)     when obtaining from other buffer must place cursor one line beneath the cyan block of missing lines
   :diffput        dp   (diff-put)        when putting to the other buffer can put cursor on the cyan block
   :diffupdate        


   vimdiff -c "windo set nofoldenable"



Remove Trailing Whitespace
----------------------------

::

   :.,34s/\s\+$//c
   :%s/\s\+$//g
   :%s/\s\+$//ge    # dont give error when not found

   :%s/\s\+$//g     # its necessary to escape the + 


Remote Editing 
----------------

* https://medium.com/usevim/vim-101-editing-remote-files-a6d2f9c8d9fb
* https://ostechnix.com/vim-tips-edit-remote-files-with-vim-on-linux/

::

    vi scp://P/.bash_profile


Deleting a range of lines without using visual selection
-----------------------------------------------------------

::

   :.,+10d        # delete the next 10 lines
   :1,1000d       # delete first 1000 lines 


Remember last edit location in file and return there on opening
----------------------------------------------------------------

* https://stackoverflow.com/questions/1682536/how-do-you-make-vim-take-you-back-where-you-were-when-you-last-edited-a-file

~/.vimrc::

    " go to the position I was when last editing the file
    au BufReadPost * if line("'\"") > 0 && line("'\"") <= line("$") | exe "normal g'\"" | endif


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

* make sure there are spaces on the last line of the block to be cut and pasted 
  that cover the entire maximum width of the block 

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



TEST else if replace
----------------------

::

    355 int spath_test::ALL()
    356 {
    357     int rc = 0 ;
    358     
    359     rc += Resolve_defaultOutputPath() ;
    360     rc += Resolve_with_undefined_token();
    361     rc += Resolve_with_undefined_TMP();
    362     rc += Resolve_inline();
    363     rc += ResolveToken(); 
    364     rc += Resolve(); 
    365     rc += Exists(); 
    366     rc += Exists2(); 
    367     rc += Basename(); 
    368     rc += Name(); 
    369     rc += Remove(); 
    370     rc += IsTokenWithFallback(); 
    371     rc += ResolveTokenWithFallback(); 
    372     rc += _ResolveToken(); 
    373     rc += Resolve(); 
    374     rc += ResolveToken1(); 
    375     rc += Resolve1(); 
    376     rc += _Check(); 
    377     rc += Write(); 
    378     
    379     return rc ;
    380 }   
    381 
    382 
    383 int spath_test::Main()
    384 {
    385     const char* TEST = ssys::getenvvar("TEST", "Resolve_defaultOutputPath" );
    386     
    387     int rc = 0 ;
    388     if(     strcmp(TEST, "Resolve_defaultOutputPath")==0 )   rc = Resolve_defaultOutputPath();
    389     else if(strcmp(TEST, "Resolve_with_undefined_token")==0) rc = Resolve_with_undefined_token();
    390     else if(strcmp(TEST, "Resolve_with_undefined_TMP")==0) Resolve_with_undefined_TMP();
    391     else if(strcmp(TEST, "Resolve_inline")==0) Resolve_inline();
    392     else if(strcmp(TEST, "ResolveToken")==0) ResolveToken();
    393     else if(strcmp(TEST, "Resolve")==0) Resolve();
    394     else if(strcmp(TEST, "Exists")==0) Exists();
    395     else if(strcmp(TEST, "Exists2")==0) Exists2();
    396     else if(strcmp(TEST, "Basename")==0) Basename();
    397     else if(strcmp(TEST, "Name")==0) Name();
    398     else if(strcmp(TEST, "Remove")==0) Remove();
    399     else if(strcmp(TEST, "IsTokenWithFallback")==0) IsTokenWithFallback();
    400     else if(strcmp(TEST, "ResolveTokenWithFallback")==0) ResolveTokenWithFallback();
    401     else if(strcmp(TEST, "_ResolveToken")==0) _ResolveToken();
    402     else if(strcmp(TEST, "Resolve")==0) Resolve();
    403     else if(strcmp(TEST, "ResolveToken1")==0) ResolveToken1();
    404     else if(strcmp(TEST, "Resolve1")==0) Resolve1();
    405     else if(strcmp(TEST, "_Check")==0) _Check();
    406     else if(strcmp(TEST, "Write")==0) Write();
    407     
    408     
    409 
    :392,408s/^\(\s*\)rc = \(\w*\)();.*$/\1else if(strcmp(TEST, "\2")==0) \2();/gc


    ## \1 : leading whitespace 
    ## \2 : method name with brackets excluded

    ## actually not quite : missed the "rc = " fixed that with below
    :.,406s/0)/0) rc =/gc



add std::setw after first stream chevron on the line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     263     std::cout 
     264         << " wavelength " << wavelength << std::endl

     263     std::cout 
     264         << std::setw(w) << " wavelength " << wavelength << std::endl

::

    :.,+20s/^\(\s*\)<</\1<< std::setw(w) <</gc

     ^\(\s*\)<<                 ## match whitespace at start of the line and capture into \1
     \1<< std::setw(w) <<       ## replace with the captured start of line and the desired suffix



    # NB: must escape the capturing bracket 


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




