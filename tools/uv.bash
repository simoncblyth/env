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

https://realpython.com/python-uv/



uv workspaces ?
-----------------

* https://docs.astral.sh/uv/concepts/projects/workspaces/

Inspired by the Cargo concept of the same name, a workspace is "a collection of
one or more packages, called workspace members, that are managed together."

Workspaces organize large codebases by splitting them into multiple packages
with common dependencies. Think: a FastAPI-based web application, alongside a
series of libraries that are versioned and maintained as separate Python
packages, all in the same Git repository.

* https://github.com/jurihock/nanobind_uv_workspace_example/blob/main/pyproject.toml


uv pip API vs uv-native pyproject.toml API
---------------------------------------------

* https://github.com/astral-sh/uv/issues/9219

The uv pip APIs are meant to resemble the pip CLI. You can think of this as a
slightly "lower-level" API: you tell uv pip to install a specific package, or
remove a specific package, and so on. The uv pip API came first, and it's
partly motivated by a desire to make it easy for folks to adopt uv without
changing their existing projects or workflows dramatically.

uv add, uv run, uv sync, and uv lock are what we call the "project APIs". These
are "higher-level": you define your dependencies in pyproject.toml, and uv
ensures that your environment is always in-sync with those dependencies.

The project APIs are more opinionated (you must use pyproject.toml, since
they're designed around "projects"), while the uv pip APIs are more flexible
(you can manipulate a virtual environment however you want -- there aren't
really any "rules"). The project APIs are more recent, and they tend to reflect
the "uv-native" workflow.

If you're starting a new project, we recommend using the project APIs. If
you're working with existing projects, it's often easier to use the uv pip
APIs, if those projects already have established workflows based on pip (since
you can just replace pip install with uv pip install, etc.).


how to use uv projects with git repo ?
-------------------------------------------

https://medium.com/@florian-trautweiler/how-i-setup-my-python-projects-with-uv-and-github-896bb8e2b184

Seems "uv init" automatically creates git repo and .venv/.gitignore contains "*"
to avoid adding dependencies to the repo. 



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
