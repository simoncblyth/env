rust-source(){   echo ${BASH_SOURCE} ; }
rust-edir(){     echo $(dirname $(rust-source)) ; }
rust-dir(){      echo $LOCAL_BASE/env/rust/rust ; }
rust-cd(){       cd $(rust-dir); }
rust-ecd(){      cd $(rust-edir); }
rust-vi(){       vi $(rust-source) ; }
rust-env(){      elocal- ; }
rust-usage(){ cat << EOU

RUST
=======

Beacuse of this glTF 2.0 viewer, am interested in rust.

* https://github.com/bwasty/gltf-viewer
* https://doc.rust-lang.org/book/2018-edition/ch01-01-installation.html

::

    epsilon:env blyth$ rust-get
    info: downloading installer

    Welcome to Rust!

    This will download and install the official compiler for the Rust programming 
    language, and its package manager, Cargo.

    It will add the cargo, rustc, rustup and other commands to Cargo's bin 
    directory, located at:

      /Users/blyth/.cargo/bin

    This path will then be added to your PATH environment variable by modifying the
    profile files located at:

      /Users/blyth/.profile
      /Users/blyth/.bash_profile

    You can uninstall at any time with rustup self uninstall and these changes will
    be reverted.

    Current installation options:

       default host triple: x86_64-apple-darwin
         default toolchain: stable
      modify PATH variable: yes

    1) Proceed with installation (default)
    2) Customize installation
    3) Cancel installation
    >
    info: syncing channel updates for 'stable-x86_64-apple-darwin'
    info: latest update on 2018-08-02, rust version 1.28.0 (9634041f0 2018-07-30)
    info: downloading component 'rustc'
     57.1 MiB /  57.1 MiB (100 %)   1.1 MiB/s ETA:   0 s                
    info: downloading component 'rust-std'
     46.8 MiB /  46.8 MiB (100 %)   1.1 MiB/s ETA:   0 s                
    info: downloading component 'cargo'
      3.1 MiB /   3.1 MiB (100 %) 981.0 KiB/s ETA:   0 s                
    info: downloading component 'rust-docs'
      9.4 MiB /   9.4 MiB (100 %)   1.1 MiB/s ETA:   0 s                
    info: installing component 'rustc'
    info: installing component 'rust-std'
    info: installing component 'cargo'
    info: installing component 'rust-docs'
    info: default toolchain set to 'stable'

      stable installed - rustc 1.28.0 (9634041f0 2018-07-30)


    Rust is installed now. Great!

    To get started you need Cargo's bin directory ($HOME/.cargo/bin) in your PATH 
    environment variable. Next time you log in this will be done automatically.

    To configure your current shell run source $HOME/.cargo/env
    epsilon:rust blyth$ cd

::

    epsilon:~ blyth$ which rustc
    /Users/blyth/.cargo/bin/rustc
    epsilon:~ blyth$ rustc --version
    rustc 1.28.0 (9634041f0 2018-07-30)
    epsilon:~ blyth$ 



EOU
}
rust-get(){
   local dir=$(dirname $(rust-dir)) &&  mkdir -p $dir && cd $dir

   curl https://sh.rustup.rs -sSf | sh  
}

rust-doc()
{
   rustup doc 
}


