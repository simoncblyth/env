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




Loadsa .rb Formula
--------------------

::

    epsilon:home blyth$ ll /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core/Formula/ | wc -l 
        4526


brew update
-------------

::

    epsilon:home blyth$ brew update
    Updated 1 tap (homebrew/core).
    ==> New Formulae
    netdata
    ==> Updated Formulae
    apache-geode                convox                      gnupg                       openimageio                 skafos
    azure-cli                   davmail                     heroku                      openvdb                     sourcery
    babl                        docker                      htmldoc                     percona-server-mongodb      sox
    bazel                       docker-completion           i2p                         pick                        spigot
    bit                         docker-compose              jhipster                    pipenv                      svgcleaner
    botan                       docker-compose-completion   kops                        pqiv                        teleport
    brotli                      fits                        libspectre                  pycodestyle                 terraform
    chakra                      fn                          libzip                      qscintilla2                 txr
    cliclick                    futhark                     lmod                        saltstack                   vault
    conan                       gdcm                        mill                        scm-manager                 watch
    container-diff              ghostscript                 nss                         sdb                         yash



Grokking the ruby formula
-----------------------------

* ~/workflow/ruby/play/
* https://github.com/Homebrew/brew/blob/master/docs/Formula-Cookbook.md
* https://stackoverflow.com/questions/24476081/homebrew-formula-syntax
* https://github.com/Homebrew/legacy-homebrew/blob/master/Library/Homebrew/formula.rb

  * search for DSL (domain specific language)
  * https://stackoverflow.com/questions/2505067/class-self-idiom-in-ruby


brew list
------------

::

    epsilon:home blyth$ brew ls carthage
    /usr/local/Cellar/carthage/0.29.0/bin/carthage
    /usr/local/Cellar/carthage/0.29.0/etc/bash_completion.d/carthage
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/ (61 files)
    /usr/local/Cellar/carthage/0.29.0/share/fish/vendor_completions.d/carthage.fish
    /usr/local/Cellar/carthage/0.29.0/share/zsh/site-functions/_carthage


    epsilon:home blyth$ brew ls -v carthage
    /usr/local/Cellar/carthage/0.29.0/LICENSE.md
    /usr/local/Cellar/carthage/0.29.0/INSTALL_RECEIPT.json
    /usr/local/Cellar/carthage/0.29.0/bin/carthage
    /usr/local/Cellar/carthage/0.29.0/.brew/carthage.rb
    /usr/local/Cellar/carthage/0.29.0/etc/bash_completion.d/carthage
    /usr/local/Cellar/carthage/0.29.0/README.md
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Resources
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Resources/Info.plist
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Scripts/carthage-bash-completion
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Scripts/carthage-zsh-completion
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Scripts/carthage-fish-completion
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/CarthageKit
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Frameworks/ReactiveTask.framework/ReactiveTask
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Frameworks/ReactiveTask.framework/Resources
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Frameworks/ReactiveTask.framework/Versions/A/ReactiveTask
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Frameworks/ReactiveTask.framework/Versions/A/Resources/Info.plist
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Frameworks/ReactiveTask.framework/Versions/Current
    /usr/local/Cellar/carthage/0.29.0/Frameworks/CarthageKit.framework/Versions/A/Frameworks/libswiftAppKit.dylib
    ...


    epsilon:home blyth$ which carthage
    /usr/local/bin/carthage

    epsilon:home blyth$ ll /usr/local/bin/
    total 72
    -r-xr-xr-x   1 root   wheel  7686 Dec 20 19:54 uninstall_cuda_drv.pl
    -rw-r--r--   1 root   wheel  8232 Dec 20 19:54 .cuda_driver_uninstall_manifest_do_not_delete.txt
    -rwxr-xr-x   1 root   wheel   538 Apr 10 21:40 fuzzy_match
    -rwxr-xr-x   1 root   wheel   532 Apr 10 21:40 xcodeproj
    -rwxr-xr-x   1 root   wheel   526 Apr 10 21:40 pod
    -rwxr-xr-x   1 root   wheel   534 Apr 10 21:40 sandbox-pod
    drwxr-xr-x  23 root   wheel   736 Apr 10 22:42 ..
    lrwxr-xr-x   1 blyth  admin    28 Apr 10 22:42 brew -> /usr/local/Homebrew/bin/brew
    lrwxr-xr-x   1 blyth  admin    38 Apr 12 11:23 carthage -> ../Cellar/carthage/0.29.0/bin/carthage
    drwxrwxr-x  10 blyth  admin   320 Apr 12 11:23 .
    epsilon:home blyth$ 



EOU
}
brew-dir(){ echo $(local-base)/env/tools/tools-brew ; }
brew-cd(){  cd $(brew-dir); }
brew-mate(){ mate $(brew-dir) ; }
brew-get(){
   local dir=$(dirname $(brew-dir)) &&  mkdir -p $dir && cd $dir

}
