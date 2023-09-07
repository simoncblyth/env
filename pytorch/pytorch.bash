pytorch-env(){ echo -n ; }
pytorch-vi(){ vi $BASH_SOURCE ; }
pytorch-usage(){ cat << EOU
pytorch-usage
====================


pytorch CUDA gcc versions
-----------------------------


* https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py#L45


::

    MINIMUM_GCC_VERSION = (5, 0, 0)
    MINIMUM_MSVC_VERSION = (19, 0, 24215)

    VersionRange = Tuple[Tuple[int, ...], Tuple[int, ...]]
    VersionMap = Dict[str, VersionRange]
    # The following values were taken from the following GitHub gist that
    # summarizes the minimum valid major versions of g++/clang++ for each supported
    # CUDA version: https://gist.github.com/ax3l/9489132
    # Or from include/crt/host_config.h in the CUDA SDK
    # The second value is the exclusive(!) upper bound, i.e. min <= version < max
    CUDA_GCC_VERSIONS: VersionMap = {
        '11.0': (MINIMUM_GCC_VERSION, (10, 0)),
        '11.1': (MINIMUM_GCC_VERSION, (11, 0)),
        '11.2': (MINIMUM_GCC_VERSION, (11, 0)),
        '11.3': (MINIMUM_GCC_VERSION, (11, 0)),
        '11.4': ((6, 0, 0), (12, 0)),
        '11.5': ((6, 0, 0), (12, 0)),
        '11.6': ((6, 0, 0), (12, 0)),
        '11.7': ((6, 0, 0), (12, 0)),
    }

    MINIMUM_CLANG_VERSION = (3, 3, 0)
    CUDA_CLANG_VERSIONS: VersionMap = {
        '11.1': (MINIMUM_CLANG_VERSION, (11, 0)),
        '11.2': (MINIMUM_CLANG_VERSION, (12, 0)),
        '11.3': (MINIMUM_CLANG_VERSION, (12, 0)),
        '11.4': (MINIMUM_CLANG_VERSION, (13, 0)),
        '11.5': (MINIMUM_CLANG_VERSION, (13, 0)),
        '11.6': (MINIMUM_CLANG_VERSION, (14, 0)),
        '11.7': (MINIMUM_CLANG_VERSION, (14, 0)),
    }




pytorch with CUDA 12 
-----------------------

* https://github.com/pytorch/pytorch/issues/91122


EOU
}
