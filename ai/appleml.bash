appleml-vi(){ vi $BASH_SOURCE ; }
appleml-env(){ echo -n ; }
appleml-usage(){ cat << EOU
appleml.bash
============

* https://github.com/ml-explore/mlx
* https://ml-explore.github.io/mlx/build/html/install.html

* https://pybind11.readthedocs.io/en/stable/index.html

  * lightweight header only alternative to boost.python



MLX is an array framework for machine learning on Apple silicon, brought to you
by Apple machine learning research.

Some key features of MLX include:

Familiar APIs: MLX has a Python API that closely follows NumPy. MLX also
has a fully featured C++ API, which closely mirrors the Python API. MLX has
higher-level packages like mlx.nn and mlx.optimizers with APIs that closely
follow PyTorch to simplify building more complex models.


EOU
}
