# === func-gen- : tools/tensorflow/tf fgp tools/tensorflow/tf.bash fgn tf fgh tools/tensorflow src base/func.bash
tf-source(){   echo ${BASH_SOURCE} ; }
tf-edir(){ echo $(dirname $(tf-source)) ; }
tf-ecd(){  cd $(tf-edir); }
tf-dir(){  echo $LOCAL_BASE/env/tools/tensorflow/tf ; }
tf-cd(){   cd $(tf-dir); }
tf-vi(){   vi $(tf-source) ; }
tf-env(){  elocal- ; }
tf-usage(){ cat << EOU

TensorFlow
=============


* https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0

* https://github.com/tensorflow/tensorflow/issues/13610

@Lancerchaing Both the (current version of the) tf.data API and the old
queue-based approach place the entire input pipeline on the CPU, and the GPUs
are exercised by parts of the graph that come after the input pipeline. Since
the overheads associated with the tf.data implementation are smaller, I'd
recommend using it over the queue-based approach. Take a look at the TF
benchmarks for an example of how to use tf.data efficiently in a multi-GPU
setup.

We still plan to add support for running stages of a tf.data pipeline on GPU,
as well as easier-to-use support for prefetching to GPU memory (currently
implemented here, but not yet exposed via the API), which should help to
achieve even better performance in future.

* https://github.com/tensorflow/tensorflow/blob/926fc13f7378d14fa7980963c4fe774e5922e336/tensorflow/contrib/data/python/ops/prefetching_ops.py#L30

* tf.experimental.prefetch_to_device

* https://github.com/horovod/horovod/issues/815

* https://eng.uber.com/horovod/


:google:`training tensor flow with data generated on GPU`


* https://stackoverflow.com/questions/56778375/make-tensorflow-use-training-data-generated-on-the-fly-by-custom-cuda-routine



* https://www.tensorflow.org/guide/performance/datasets



EOU
}
tf-get(){
   local dir=$(dirname $(tf-dir)) &&  mkdir -p $dir && cd $dir

}
