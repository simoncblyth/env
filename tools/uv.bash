# === func-gen- : tools/uv fgp tools/uv.bash fgn uv fgh tools src base/func.bash
uv-source(){   echo ${BASH_SOURCE} ; }
uv-edir(){ echo $(dirname $(uv-source)) ; }
uv-ecd(){  cd $(uv-edir); }
uv-dir(){  echo $LOCAL_BASE/env/tools/uv ; }
uv-cd(){   cd $(uv-dir); }
uv-vi(){   vi $(uv-source) ; }
uv-env(){  elocal- ; }
uv-usage(){ cat << EOU


uv : faster pip/virtualenv built in rust
===========================================

https://docs.astral.sh/uv/

https://github.com/astral-sh/uv

https://www.datacamp.com/tutorial/python-uv

https://medium.com/@datagumshoe/using-uv-and-conda-together-effectively-a-fast-flexible-workflow-d046aff622f0


zeta/lch installed UV into conda home env
-------------------------------------------

DID I ?



Zeta .local/bin install
-------------------------

    zeta:~ blyth$ cat ~/.curlrc 
    proxy=socks5h://127.0.0.1:8080

    zeta:~ blyth$ curl -LsSf https://astral.sh/uv/install.sh | cat > uv_install.sh  ## save first to allow a quick check

    zeta:~ blyth$ cat uv_install.sh | sh
    downloading uv 0.8.14 aarch64-apple-darwin
    no checksums to verify
    installing to /Users/blyth/.local/bin
      uv
      uvx
    everything's installed!

    To add $HOME/.local/bin to your PATH, either restart your shell or run:

        source $HOME/.local/bin/env (sh, bash, zsh)
        source $HOME/.local/bin/env.fish (fish)

    zeta:~ blyth$ 
    zeta:~ blyth$ l .local/bin/
    total 77928
        8 -rw-r--r--  1 blyth  staff       165 Sep  3 17:08 env.fish
        0 drwxr-xr-x  6 blyth  staff       192 Sep  3 17:08 .
        8 -rw-r--r--  1 blyth  staff       328 Sep  3 17:08 env
        0 drwxr-xr-x  3 blyth  staff        96 Sep  3 17:08 ..
    77248 -rwxr-xr-x  1 blyth  staff  39546960 Aug 29 05:51 uv
      664 -rwxr-xr-x  1 blyth  staff    336480 Aug 29 05:51 uvx
    zeta:~ blyth$ 

    zeta:~ blyth$ cat $HOME/.local/bin/env
    #!/bin/sh
    # add binaries to PATH if they aren't added yet
    # affix colons on either side of $PATH to simplify matching
    case ":${PATH}:" in
        *:"$HOME/.local/bin":*)
            ;;
        *)
            # Prepending path in case a system-installed binary needs to be overridden
            export PATH="$HOME/.local/bin:$PATH"
            ;;
    esac
    zeta:~ blyth$ 

The installer appended to .bash_profile::

     82 
     83 
     84 . "$HOME/.local/bin/env"





EOU
}
uv-get(){
   local dir=$(dirname $(uv-dir)) &&  mkdir -p $dir && cd $dir

}
