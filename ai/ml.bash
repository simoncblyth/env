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


EOU
}
ml-get(){
   local dir=$(dirname $(ml-dir)) &&  mkdir -p $dir && cd $dir
}

ml-sutton-url(){ echo  http://incompleteideas.net/book/bookdraft2017nov5.pdf ; }
ml-sutton(){ open $(ml-dir)/sutton-bookdraft2017nov5.pdf ; }


