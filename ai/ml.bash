ml-source(){   echo ${BASH_SOURCE} ; }
ml-edir(){ echo $(dirname $(ml-source)) ; }
ml-ecd(){  cd $(ml-edir); }
ml-dir(){  echo ${ML_DIR:-$LOCAL_BASE/env/ai/ml} ; }
ml-cd(){   cd $(ml-dir); }
ml-vi(){   vi $(ml-source) ; }
ml-env(){  elocal- ; }
ml-usage(){ cat << EOU

Machine Learning : Place to collect some references
=========================================================



PyTorch
---------

* https://pytorch.org/
* https://pytorch.org/tutorials/

* https://deepsense.ai/keras-or-pytorch/

..preferred solution for academic research, and applications of deep learning
requiring optimizing custom expressions. It’s supported by Facebook.


* http://www.goldsborough.me/ml/ai/python/2018/02/04/20-17-20-a_promenade_of_pytorch/

* https://wrosinski.github.io/deep-learning-frameworks/

  PyTorch and TensorFlow similar performance, Keras a 1.5-2x slower.


* https://medium.com/@iliakarmanov/multi-gpu-rosetta-stone-d4fa96162986


NVIDIA DALI
-------------

* https://github.com/NVIDIA/dali

NVIDIA Data Loading Library (DALI) is a collection of highly optimized building
blocks, and an execution engine, to accelerate the pre-processing of the input
data for deep learning applications. DALI provides both the performance and the
flexibility for accelerating different data pipelines as a single library. This
single library can then be easily integrated into different deep learning
training and inference applications.



Source for papers
-------------------

* https://openreview.net/search?term=spherical&content=all&group=all&source=all


Gauge Equivariant Convolutional Networks and the Icosahedral CNN

* https://www.youtube.com/watch?v=wZWn7Hm8osA
* https://towardsdatascience.com/an-easy-guide-to-gauge-equivariant-convolutional-networks-9366fb600b70



Geometric Deep Learning
------------------------

* http://geometricdeeplearning.com/

* https://www.youtube.com/watch?v=D3fnGG7cdjY

  Geometric Deep Learning

* GCN : Graph Convolutional Networks

* https://github.com/search?q=GCN


GCN
----

* https://github.com/tkipf/gcn
* http://tkipf.github.io/graph-convolutional-networks/


Representing Graphs
----------------------

* https://www.youtube.com/watch?v=9C2cpQZVRBA&vl=en
 
  Adjacency Matrix : V^2 : often impractical 

* https://www.youtube.com/watch?v=k1wraWzqtvQ

  Adjacency List  




GNN : Graph Neural Networks
-------------------------------

* Graph neural networks: Variations and applications

* Represented with adjacency matrix





CNN, Convolutional Neural Nets, ConvNets
------------------------------------------

* https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

* http://cs231n.stanford.edu/

  CS231n: Convolutional Neural Networks for Visual Recognition, Spring 2019
 
* http://cs231n.github.io/convolutional-networks/


:google:`SphereNet CNN`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/ChiWeiHsiao/SphereNet-pytorch

unofficial implementation of ECCV 18 paper "SphereNet: Learning Spherical Representations for Detection and Classification in Omnidirectional Images"


* https://www.semanticscholar.org/paper/SphereNet%3A-Learning-Spherical-Representations-for-Coors-Condurache/8a0cb93b7a4e0cfe6c8cb6e5cdd0bc199b515ebc


Learning SO(3) Equivariant Representations with Spherical CNNs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.youtube.com/watch?v=Y86rzE4UzKs
* https://machc.github.io/  Carlos Esteves


DeepSphere
~~~~~~~~~~~~~

* https://github.com/SwissDataScienceCenter/DeepSphere
* https://zenodo.org/record/3243381#.XSBERnsRXOQ

* https://github.com/mdeff/cnn_graph
* https://github.com/epfl-lts2/pygs



s2cnn : Spherical CNNs
~~~~~~~~~~~~~~~~~~~~~~~~~

* https://openreview.net/forum?id=Hkbd5xZRb

* https://github.com/jonas-koehler/s2cnn
* http://pytorch.org/
* https://github.com/cupy/cupy
  
  CuPy is an implementation of NumPy-compatible multi-dimensional array on CUDA. 

* https://github.com/AMLab-Amsterdam/lie_learn
* https://github.com/NVIDIA/pynvrtc


Spherical CNNs on Unstructured Grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chiyu Max Jiang, Jingwei Huang, Karthik Kashinath, Prabhat, Philip Marcus, Matthias Niessner

* https://arxiv.org/abs/1901.02039
* https://openreview.net/forum?id=Bkl-43C9FQ


* https://github.com/maxjiang93/ugscnn

* https://libigl.github.io/


Swiss Data Science Center
----------------------------

* https://github.com/SwissDataScienceCenter




Competitions
-------------

* https://www.kaggle.com/competitions
* https://www.import.io/post/how-to-win-a-kaggle-competition/

It used to be random forest that was the big winner, but over the last six
months a new algorithm called XGboost has cropped up, and it’s winning
practically every competition in the structured data category.

For any dataset that contains images or speech problems, deep learning is the
way to go. The people who are winning these competitions (the ones without
well-structured data) are spending almost none of their time doing feature
engineering. Instead, they spend their time constructing neural networks.


* https://www.kaggle.com/c/flavours-of-physics-kernels-only/overview/description
* https://github.com/yandexdataschool/flavours-of-physics-start


* https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only/discussion/64739
* https://www.kaggle.com/docs/competitions#kernels-only-FAQ

* https://www.kaggle.com/docs/competitions#resources-for-getting-started

* https://www.kaggle.com/c/titanic

* https://towardsdatascience.com/introduction-to-kaggle-kernels-2ad754ebf77

* https://www.kaggle.com/learn/overview



* https://www.kaggle.com/c/trackml-particle-identification



Approximate Bayesian computation (ABC) 
----------------------------------------

* :google:`likelihood free inference`


Libs
------

* https://skymind.ai/wiki/comparison-frameworks-dl4j-tensorflow-pytorch

Pytorch
    open-sourced by Facebook in January 2017
    dynamic computation graphs, which let you process variable-length inputs and outputs, 
    which is useful when working with RNNs, for example



Particle physics in the era of Artificial Intelligence
---------------------------------------------------------

* Kyle Stuart Cranmer (New York University (US))

* https://indico.cern.ch/event/666278/contributions/2830616/
* https://indico.cern.ch/event/666278/contributions/2830616/attachments/1579293/2495102/Skeikampen-Physics-AI.pdf


* https://github.com/cranmer/active_sciencing

  Interesingly, we will use the simulator not only to perform inference on the
  parameters, but also to design the next experiment (this is where active
  learning comes in). 


* https://github.com/cranmer/active_sciencing/blob/master/demo_gaussian.ipynb



likelihood-free inference (LFI) 
----------------------------------

* :google:`likelihood-free inference`

* https://elfi.readthedocs.io/en/latest/



Reinforcement Learning
------------------------

* https://towardsdatascience.com/model-based-reinforcement-learning-cb9e41ff1f0d


Libs
-------

* https://www.h2o.ai/products/h2o4gpu/


Gradient Boosting
--------------------

* homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
* https://devblogs.nvidia.com/gradient-boosting-decision-trees-xgboost-cuda/


Deep Learning Reconstruction
-------------------------------

* http://inspirehep.net/record/1686986/
* https://pos.sissa.it/301/1057

  Deep Learning in Physics exemplified by the Reconstruction of Muon-Neutrino Events in IceCube
   


* http://deeplearnphysics.org/#introduction

  TPC  

* https://arxiv.org/abs/1611.05531

  Convolutional Neural Networks Applied to Neutrino Events in a Liquid Argon Time Projection Chamber (MicroBooNE)


* https://www.hindawi.com/journals/ahep/2018/7024309/

  Deep Learning the Effects of Photon Sensors on the Event Reconstruction Performance in an Antineutrino Detector



Dynamic Deep Learning
------------------------

* https://medium.com/@Petuum/intro-to-dynamic-neural-networks-and-dynet-67694b18cb23

  Intro to Dynamic Neural Networks and DyNet




:google:`machine learning together with fast monte carlo generation`
-----------------------------------------------------------------------


OCT reconstruction using a 10,000x faster MC : Zhao, Sinan Thesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://thesis.library.caltech.edu/9597/

  Zhao, Sinan (2016) 
  Advanced Monte Carlo Simulation and Machine Learning for Frequency Domain Optical Coherence Tomography. 
  Dissertation (Ph.D.), California Institute of Technology. doi:10.7907/Z9X63JVM

  Explains use of 10,000x faster MC to train in the reconstruction of truth for OCT images


p35

We then build a hierarchy architecture of machine learning models (committee
of experts) based on extremely randomized trees (extra trees), and train
different parts of the architecture with specifically designed data sets.

In prediction, an unseen OCT image first goes through a classification model to
determine its structure (e.g., the number and the types of layers present in
the image); then the image is handed to a regression model that is trained
specif- ically for that particular structure to predict the length of the
different layers and by doing so reconstruct the ground-truth of the image.

p71  Solving the learning problem

p90 Classification/Committee of experts

    Training each class with a dedicated sample



Reinforcement Learning: Introduction to Monte Carlo Learning using the OpenAI Gym Toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://medium.com/deep-math-machine-learning-ai/ch-12-reinforcement-learning-complete-guide-towardsagi-ceea325c5d53

* https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/







Stanford CS229 : Machine Learning
----------------------------------

* http://cs229.stanford.edu/syllabus.html


Intro http://cs229.stanford.edu/notes-spring2019/lecture1_slide.pdf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Taxonomy of machine learning

supervised 
    training set (input objects, eg PMT signals)
    output object "labels" (eg MC truth muon parameters)  

    * unambiguous "right answer"

unsupervised
    no labels (eg clustering)

reinforcement
   feedback loop of data collection and training 

   * http://cs229.stanford.edu/notes-spring2019/cs229-notes12.pdf

   * do not know the "right answer" eg teach a four legged robot to walk
     so cannot provide explicit supervision 
 
   * often uses MDP (Markov Decision Processes)


reinforcement learning and MDPs
----------------------------------

* :google:`reinforcement learning MDPs`

* https://towardsdatascience.com/reinforcement-learning-demystified-markov-decision-processes-part-1-bf00dda41690?gi=493bbbf3c43

The Markov property states that, "The future is independent of the past given the present."

* ie everything is in the state : can discard the past 


ml-sutton
~~~~~~~~~~

* http://incompleteideas.net/book/bookdraft2017nov5.pdf 

  Reinforcement Learning: An Introduction
  Richard S. Sutton and Andrew G. Barto (November 5, 2017)
  445 pages  

p1,2 
    good introduction to differences between 
    supervised/unsupervised/reinforcement learning 

GANs
-----

* https://skymind.ai/wiki/generative-adversarial-network-gan
* http://cs229.stanford.edu/notes/cs229-notes2.pdf



XGBoost : optimized distributed gradient boosting library
------------------------------------------------------------

bk-chollet:

   ..two techniques you should be the most familiar with in order to be
   successful in applied machine learning today: gradient boosting machines, for
   shallow- learning problems; and deep learning, for perceptual problems.

* https://xgboost.readthedocs.io/en/latest/
* https://xgboost.readthedocs.io/en/latest/tutorials/model.html



EOU
}
ml-get(){
   local dir=$(dirname $(ml-dir)) &&  mkdir -p $dir && cd $dir
}

ml-sutton-url(){ echo  http://incompleteideas.net/book/bookdraft2017nov5.pdf ; }
ml-sutton(){ open $(ml-dir)/sutton-bookdraft2017nov5.pdf ; }


