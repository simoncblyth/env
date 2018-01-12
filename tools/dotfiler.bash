# === func-gen- : tools/dotfiler env fgp tools/dotfiler.bash fgn dotfiler fgh tools
dotfiler-src(){      echo tools/dotfiler.bash ; }
dotfiler-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dotfiler-src)} ; }
dotfiler-vi(){       vi $(dotfiler-source) ; }
dotfiler-env(){      elocal- ; }
dotfiler-usage(){ cat << EOU

dotfiler
===========

Install
---------

::

   dotfiler-get


Usage
-------

::

    delta:~ blyth$ dot --help
    Dotfiles manager

    Usage:
      dot update [--dry] [--verbose] [--base-dir=<base-dir>] [--home-dir=<home-dir>] [--skip-pull]
      dot status [--base-dir=<base-dir>]
      dot add [--base-dir=<base-dir>] [--verbose] <url>...
      dot (-h | --help)
      dot --version

    Options:
      -h --help                Show this screen.
      --version                Show version.
      -v --verbose             More verbose output.
      --dry                    Don't make real modification, just print what will be done.
      --base-dir=<base-dir>    Directory to search environments [default: /Users/blyth/.dotfiler].
      --home-dir=<home-dir>    Directory, where files should be linked to [default: /Users/blyth].



Workflow
---------

*add* 
    git clones dot-envname repo urls such as the below into base-dir 
    https://github.com/svetlyak40wt/dot-emacs

*status*
    for environments (non-ignored dirs in base-dir with .git) dump the git status

*update*
    without --skip-pull pulls any remote updates into git repo 
      


Test::

    delta:~ blyth$ which dot
    /Users/blyth/.dotfiler/bin/dot

    delta:~ blyth$ dot add svetlyak40wt/dot-emacs --verbose
    INFO    Cloning repository "https://github.com/svetlyak40wt/dot-emacs to "emacs" dir.
    Cloning into 'emacs'...
    remote: Counting objects: 390, done.
    remote: Total 390 (delta 0), reused 0 (delta 0), pack-reused 390
    Receiving objects: 100% (390/390), 126.67 KiB | 0 bytes/s, done.
    Resolving deltas: 100% (164/164), done.
    Checking connectivity... done.

    delta:~ blyth$ dot add svetlyak40wt/dot-emacs --verbose
    ERROR   Environment "emacs" already exists.


    delta:~ blyth$ l .dotfiler/
    total 32
    drwxr-xr-x  6 blyth  staff    204 Jan 11 18:54 emacs
    -rw-r--r--  1 blyth  staff   1799 Jan 11 18:53 CHANGELOG.md
    -rw-r--r--  1 blyth  staff  10868 Jan 11 18:53 README.md
    drwxr-xr-x  4 blyth  staff    136 Jan 11 18:53 bin

    delta:~ blyth$ dot --dry update
    LINK    Symlink from  /Users/blyth/.emacs.d to /Users/blyth/.dotfiler/emacs/.emacs.d will be created

    delta:~ blyth$ dot -v update
    INFO    Making pull in "emacs":
    INFO        Already up-to-date.
    LINK    Symlink from /Users/blyth/.emacs.d to /Users/blyth/.dotfiler/emacs/.emacs.d was created


    delta:.emacs.d blyth$ git status
    On branch master
    Your branch is up-to-date with 'origin/master'.

    nothing to commit, working directory clean

    delta:.emacs.d blyth$ ls -alst    ## huh, where is the .git dir ?
    total 32
     0 drwxr-xr-x   7 blyth  staff   238 Jan 11 19:02 .
     0 drwxr-xr-x   6 blyth  staff   204 Jan 11 19:02 ..
     8 -rw-r--r--   1 blyth  staff   184 Jan 11 19:02 .gitignore
    16 -rw-r--r--   1 blyth  staff  4378 Jan 11 19:02 customizations.el
     8 -rw-r--r--   1 blyth  staff  4003 Jan 11 19:02 init.el
     0 drwxr-xr-x   9 blyth  staff   306 Jan 11 19:02 lib
     0 drwxr-xr-x  19 blyth  staff   646 Jan 11 19:02 old

    delta:.dotfiler blyth$ cd emacs/   ## .git is one level up, providing a place for a README
    delta:emacs blyth$ ll
    total 16
    -rw-r--r--   1 blyth  staff   573 Jan 11 19:02 README.md
    -rw-r--r--   1 blyth  staff  1549 Jan 11 19:02 ChangeLog.md
    drwxr-xr-x   7 blyth  staff   238 Jan 11 19:02 .emacs.d
    drwxr-xr-x   6 blyth  staff   204 Jan 11 19:02 .
    drwxr-xr-x  10 blyth  staff   340 Jan 11 19:07 ..
    drwxr-xr-x  15 blyth  staff   510 Jan 11 19:09 .git





https://github.com/svetlyak40wt/dotfiler

core.py *add* command assumes github::

    496 def _normalize_url(url):
    497     """Returns tuple (real_url, env_name), using
    498     following rules:
    499     - if url has scheme, its returned as is.
    500     - if url is in the form username/repo, then
    501       we consider they are username/repo at the github
    502       and return full https url.
    503     - env_name is a last part of the path with removed
    504       '.git' suffix and 'dot[^-]*-' prefix.
    505     """
    506 
    507     # extract name
    508     name = url.rsplit('/', 1)[-1]
    509     name = re.sub(r'^dot[^-]*-', '', name)
    510     name = re.sub(r'\.git$', '', name)
    511 
    512     # check if this is a github shortcut
    513     match = re.match('^([^/:]+)/([^/]+)$', url)
    514     if match is not None:
    515         url = 'https://github.com/' + url
    516     return (url, name)



EOU
}

dotfiler-dir(){ echo $HOME/.dotfiler ; }
dotfiler-cd(){  cd $(dotfiler-dir); }

dotfiler-wipe(){ rm -rf $(dotfiler-dir) ; }

dotfiler-get(){
   local dir=$(dirname $(dotfiler-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(basename $(dotfiler-dir))

   [ ! -d $nam ] && git clone https://github.com/svetlyak40wt/dotfiler $nam

   dotfiler-hookup
}


dotfiler-hookup(){ cat << EOI

Add below line to .bash_profile for hookup

  export PATH=\$(dotfiler-dir)/bin:\$PATH

EOI
}





