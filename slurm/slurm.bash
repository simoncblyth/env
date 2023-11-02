slurm-env(){ echo -n ; }
slurm-vi(){ vi $BASH_SOURCE ; }
slurm-usage(){ cat << EOU
slurm.bash : sbatch/srun/squeue
==================================

* https://slurm.schedmd.com/documentation.html

Submit jobs for GPU cluster from L7 with sbatch 
and interactive tests with srun 

See also:

~/j/lxslc7.bash 
    submit funcs etc...

~/j/gpujob.bash 
    bash carrier job 

EOU
}

slurm-bash()
{
   : ~/env/slurm/slurm.bash from Tao : slurm-run-interactive
   type $FUNCNAME
   [ -n "$NODE" ] && opt="$opt -w $NODE"   # eg NODE=gpu015
   [ -n "$NCORES" ] && opt="$opt --cpus-per-task=$NCORES";
   local cmd="srun $opt --pty -p gpu --gres=gpu:1 bash"
   echo $cmd
   eval $cmd
}


