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


Regexp Replace
---------------

For example turning an enum into a switch statement::

    .,+20s/\s*\(\S*\).*/case \1 : s="\1" ;break; /gc


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




