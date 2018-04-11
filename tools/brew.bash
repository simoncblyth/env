# === func-gen- : tools/brew fgp tools/brew.bash fgn brew fgh tools
brew-src(){      echo tools/brew.bash ; }
brew-source(){   echo ${BASH_SOURCE:-$(env-home)/$(brew-src)} ; }
brew-vi(){       vi $(brew-source) ; }
brew-env(){      elocal- ; }
brew-usage(){ cat << EOU

Brew
=======

* https://brew.sh/
* ~/tree/brew/brew.log 

Oh my its messy::

    epsilon:~ blyth$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    ==> This script will install:
    /usr/local/bin/brew
    /usr/local/share/doc/homebrew
    /usr/local/share/man/man1/brew.1
    /usr/local/share/zsh/site-functions/_brew
    /usr/local/etc/bash_completion.d/brew
    /usr/local/Homebrew
    ==> The following existing directories will be made group writable:
    /usr/local/bin
    ==> The following existing directories will have their owner set to blyth:
    /usr/local/bin
    ==> The following existing directories will have their group set to admin:
    /usr/local/bin
    ==> The following new directories will be created:
    /usr/local/Cellar
    /usr/local/Homebrew
    /usr/local/Frameworks
    /usr/local/etc
    /usr/local/include
    /usr/local/lib
    /usr/local/opt
    /usr/local/sbin
    /usr/local/share
    /usr/local/share/zsh
    /usr/local/share/zsh/site-functions
    /usr/local/var

    Press RETURN to continue or any other key to abort

::

    epsilon:~ blyth$ brew info wget 
    wget: stable 1.19.4 (bottled), HEAD
    Internet file retriever
    https://www.gnu.org/software/wget/
    Not installed
    From: https://github.com/Homebrew/homebrew-core/blob/master/Formula/wget.rb
    ==> Dependencies
    Build: pkg-config ✘
    Required: libidn2 ✘, openssl ✘
    Optional: pcre ✘, libmetalink ✘, gpgme ✘
    ==> Options
    --with-debug
        Build with debug support
    --with-gpgme
        Build with gpgme support
    --with-libmetalink
        Build with libmetalink support
    --with-pcre
        Build with pcre support
    --HEAD
        Install HEAD version
    epsilon:~ blyth$ 



Grokking the ruby formula
-----------------------------

* ~/workflow/ruby/play/
* https://github.com/Homebrew/brew/blob/master/docs/Formula-Cookbook.md
* https://stackoverflow.com/questions/24476081/homebrew-formula-syntax
* https://github.com/Homebrew/legacy-homebrew/blob/master/Library/Homebrew/formula.rb

  * search for DSL (domain specific language)
  * https://stackoverflow.com/questions/2505067/class-self-idiom-in-ruby




EOU
}
brew-dir(){ echo $(local-base)/env/tools/tools-brew ; }
brew-cd(){  cd $(brew-dir); }
brew-mate(){ mate $(brew-dir) ; }
brew-get(){
   local dir=$(dirname $(brew-dir)) &&  mkdir -p $dir && cd $dir

}
