#!/usr/bin/env bash
usage(){ cat << EOU
cuml_check.sh : rapidsai/cuml, ipython, pyvista
=================================================

~/env/tools/cuml_check/cuml_check.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

. $HOME/conda_base.sh && conda_base

#conda_env=cuml_env_py310        # runtime SEGV
conda_env=cuml_env_py312         # WORKS but has lots of build warnings
#conda_env=cuml_env              # WORKS but gives py313 build warnings

conda activate ${conda_env}

#script=pointcloud_nearest_neighbors.py
#script=pointcloud_overlap_2D.py
script=pointcloud_overlap_3D.py

defarg="info_list_pdb"
arg=${1:-$defarg}

vv="BASH_SOURCE defarg arg PWD conda_env script"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/list}" != "$arg" ]; then
    conda list | grep -E 'vtk|pyvista|cuml|libexpat|libsqlite|expat|sqlite|librmm|libkvikio|libcudf|libraft|libcurand|rapids|cudatoolkit'
fi


if [ "${arg/pdb}" != "$arg" ]; then
    which ipython
    cmd="ipython --pdb -i $script"
    echo $cmd
    eval $cmd
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 1
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    gdb -ex r --args $(which python) $PWD/$script
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

exit 0


