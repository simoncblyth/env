# === func-gen- : base/vim/vim fgp base/vim/vim.bash fgn vim fgh base/vim
vim-src(){      echo base/vim/vim.bash ; }
vim-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vim-src)} ; }
vim-vi(){       vi $(vim-source) ; }
vim-env(){      elocal- ; }
vim-usage(){ cat << EOU

VIM Tips
=========


* http://www.astrohandbook.com/ch20/vi_guide.html


Gutter Line Numbers
----------------------

* http://vim.wikia.com/wiki/Display_line_numbers


Regexp Replace
---------------

For example turning an enum into a switch statement::

    .,+50s/\s*case \(\S*\):/case \1:  s="\1" ;break;/gc



Overwrite Mode : handy for ascii art
--------------------------------------

* shift-R



EOU
}
vim-dir(){ echo $(local-base)/env/base/vim/base/vim-vim ; }
vim-cd(){  cd $(vim-dir); }
vim-mate(){ mate $(vim-dir) ; }
vim-get(){
   local dir=$(dirname $(vim-dir)) &&  mkdir -p $dir && cd $dir

}
