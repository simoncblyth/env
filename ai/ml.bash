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


