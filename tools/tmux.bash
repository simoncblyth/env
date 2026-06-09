tmux-env(){ echo -n ; }
tmux-vi(){ vi $BASH_SOURCE ; }
tmux-usage(){ cat << EOU
tmux : terminal multiplexer
============================

* https://www.redhat.com/en/blog/introduction-tmux-linux




Every tmux command starts with the Prefix key, which by default is Ctrl + b.
You press that, let go, and then press your action key::


   Split Vertically   (side-by-side)  Ctrl + b then %
   Split Horizontally (top/bottom)    Ctrl + b then "
   Move between panels                Ctrl + b then Arrow Keys
   Close current panel                Just type exit or Ctrl + d
   Toggle zooming to show one panel   Ctrl + b then z          [THIS IS HANDY WHEN COPY/PASTE]

   Scroll back/forward panel          Ctrl + b then [          [THIS ENTERS COPY MODE]
   Exit copy mode                     just press q


When using srun to get access to a GPU node eg with::

    oj6k () 
    { 
        : bash session on the server - eg to check nvidia-smi CUDA version etc;
        export TMP=$HOME/tmp
        export TEST=medium_scan
        srun --partition=junogpu --qos=junoatmgpu --gres=gpu:pro6000:1 --cpus-per-task=1 --mem=4G --pty bash
    }

Try direct tmux::

    oj6t()
    {
       : ~/oj/oj.bash
       : tmux building and testing
       oj6_env
       srun --partition=junogpu --qos=junoatmgpu --gres=gpu:pro6000:1 --cpus-per-task=8 --mem=16G --pty tmux new-session -A -s opticks_work
    }




It means that the only connection to that machine is through the one session
with no easy way to have a separate session. So have to multiplex, making 
tmux handy for just this situation. This allows splitting the terminal
into two panels with nvtop running in one and a GPU using script
run in the other.


EOU
}
