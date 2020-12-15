cel-source(){   echo ${BASH_SOURCE} ; }
cel-edir(){ echo $(dirname $(cel-source)) ; }
cel-ecd(){  cd $(cel-edir); }
cel-dir(){  echo $LOCAL_BASE/env/gpuhep/celeritas ; }
cel-cd(){   cd $(cel-dir); }
cel-vi(){   vi $(cel-source) ; }
cel-env(){  elocal- ; }
cel-usage(){ cat << EOU

Celeritas—a nascent GPU detector simulation code
==================================================

ORNL (Oak Ridge National Lab), Seth R Johnson

https://github.com/celeritas-project/celeritas

https://github.com/celeritas-project/celeritas/wiki/Development

https://www.snowmass21.org/docs/files/summaries/CompF/SNOWMASS21-CompF2_CompF1-053.pdf

https://www.youtube.com/watch?v=GcNDs0IY0vY


std::span
---------

* https://stackoverflow.com/questions/45723819/what-is-a-span-and-when-should-i-use-one


src/sim/Types.hh

     16 using EventId = OpaqueId<struct Event>;
     17 using TrackId = OpaqueId<struct Track>;


     23  * \tparam Instantiator Class that uses the indexing type.
     24  * \tparam T Value type for the ID.
     25  *
     26  * This allows type-safe, read-only indexing/access for a class. The value is
     27  * 'true' if it's assigned, 'false' if invalid.
     28  */
     29 template<class Instantiator, class T = unsigned int>
     30 class OpaqueId
     31 {
     32   public:
     33     //@{
     34     //! Type aliases
     35     using instantiator_type = Instantiator;
     36     using value_type        = T;
     37     //@}


::

     35 struct KNDemoResult
     36 {
     37     using size_type = celeritas::size_type;
     38 
     39     std::vector<double>    time;  //!< Real time per step
     40     std::vector<size_type> alive; //!< Num living tracks per step
     41     std::vector<double>    edep;  //!< Energy deposition along the grid
     42     double                 total_time = 0; //!< All time
     43 };



/usr/local/env/gpuhep/celeritas/app/demo-interactor/KNDemoKernel.cu

* https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/



::

     36 __global__ void initialize_kn(ParamPointers const   params,
     37                               StatePointers const   states,
     38                               InitialPointers const init)
     39 {   
     40     // Grid-stride loop, see
     41     for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
     42          tid < static_cast<int>(states.size());
     43          tid += blockDim.x * gridDim.x)
     44     {   
     45         ParticleTrackView particle(
     46             params.particle, states.particle, ThreadId(tid));

     ///  sets particle._state to the appropriate for the tid

     47         particle = init.particle;

     /// changes particle._state    (HUH: all the same, no tid ?) 

     48 

     43 //! Pointers to initial conditoins
     44 struct InitialPointers
     45 {   
     46     celeritas::ParticleTrackState particle;
     47 };       


Curious overloaded "operator=" for cross-type initialization from ParticleTrackState to ParticleTrackView
that just sets the state_ without touching params_.

src/physics/base/ParticleTrackView.hh::

     29 class ParticleTrackView
     30 {
     31   public:
     32     //@{
     33     //! Type aliases
     34     using Initializer_t = ParticleTrackState;
     35     //@}
     36 
     37   public:
     38     // Construct from "dynamic" state and "static" particle definitions
     39     inline CELER_FUNCTION
     40     ParticleTrackView(const ParticleParamsPointers& params,
     41                       const ParticleStatePointers&  states,
     42                       ThreadId                      id);
     43 
     44     // Initialize the particle
     45     inline CELER_FUNCTION ParticleTrackView&
     46                           operator=(const Initializer_t& other);
     47 
     ..
     84   private:
     85     const ParticleParamsPointers& params_;
     86     ParticleTrackState&           state_;
     87 


src/physics/base/ParticleTrackView.i.hh::

     20 CELER_FUNCTION
     21 ParticleTrackView::ParticleTrackView(const ParticleParamsPointers& params,
     22                                      const ParticleStatePointers&  states,
     23                                      ThreadId                      id)
     24     : params_(params), state_(states.vars[id.get()])
     25 {
     26     REQUIRE(id < states.vars.size());
     27 }
     ..
     33 CELER_FUNCTION ParticleTrackView&
     34 ParticleTrackView::operator=(const Initializer_t& other)
     35 {
     36     REQUIRE(other.def_id < params_.defs.size());
     37     REQUIRE(other.energy > zero_quantity());
     38     state_ = other;
     39     return *this;
     40 }


src/physics/base/SecondaryAllocatorView.hh::

    using SecondaryAllocatorView = StackAllocatorView<Secondary>;

src/physics/em/KleinNishinaInteractor.i.hh::

     45 template<class Engine>
     46 CELER_FUNCTION Interaction KleinNishinaInteractor::operator()(Engine& rng)
     47 {
     48     // Allocate space for the single electron to be emitted
     49     Secondary* electron_secondary = this->allocate_(1);
     50     if (electron_secondary == nullptr)
     51     {   
     52         // Failed to allocate space for a secondary
     53         return Interaction::from_failure();
     54     }


WOW : parallel dynamic allocation globally serialized with atomic_add src/base/StackAllocatorView.i.hh::

     25 //---------------------------------------------------------------------------//
     26 /*!
     27  * Allocate space for a given number of itemss.
     28  *
     29  * Returns NULL if allocation failed due to out-of-memory. Ensures that the
     30  * shared size reflects the amount of data allocated
     31  */
     32 template<class T>
     33 CELER_FUNCTION auto StackAllocatorView<T>::operator()(size_type count)
     34     -> result_type
     35 {   
     36     static_assert(std::is_default_constructible<T>::value,
     37                   "Value must be default constructible");
     38     
     39     // Atomic add 'count' to the shared size
     40     size_type start = atomic_add(shared_.size, count);
     41     if (CELER_UNLIKELY(start + count > shared_.storage.size()))
     42     {   
     43         // Out of memory: restore the old value so that another thread can
     44         // potentially use it. Multiple threads are likely to exceed the
     45         // capacity simultaneously. Only one has a "start" value less than or
     46         // equal to the total capacity: the remainder are (arbitrarily) higher
     47         // than that.
     48         if (start <= this->capacity())
     49         {   
     50             // We were the first thread to exceed capacity, even though other
     51             // threads might have failed (and might still be failing) to
     52             // allocate. Restore the actual allocated size to the start value.
     53             // This might allow another thread with a smaller allocation to
     54             // succeed, but it also guarantees that at the end of the kernel,
     55             // the size reflects the actual capacity.
     56             *shared_.size = start;
     57         }
     58         
     59         // TODO It might be useful to set an "out of memory" flag to make it
     60         // easier for host code to detect whether a failure occurred, rather
     61         // than looping through primaries and testing for failure.
     62         
     63         // Return null pointer, indicating failure to allocate.
     64         return nullptr;
     65     }
     66     
     67     // Initialize the data at the newly "allocated" address
     68     value_type* result = new (shared_.storage.data() + start) value_type;
     69     for (size_type i = 1; i < count; ++i)
     70     {
     71         // Initialize remaining values
     72         new (shared_.storage.data() + start + i) value_type;
     73     }
     74     return result;
     75 }


:google:`CUDA C++ placement new`

* https://arxiv.org/pdf/1908.05845.pdf

  Memory-Efficient Object-Oriented Programming on GPUs by Matthias SPRINGER

  ~/opticks_refs/Memory_Efficient_OOP_on_GPUs_Matthias_Springer_1908.05845.pdf

* https://www.gcc.tu-darmstadt.de/media/gcc/papers/Widmer_2013_FDM.pdf

  Fast Dynamic Memory Allocator for Massively Parallel Architectures









EOU
}


cel-url(){ echo https://github.com/celeritas-project/celeritas ; }

cel-get(){
   local dir=$(dirname $(cel-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d celeritas ] && git clone $(cel-url) 

}

cel-f(){ find . -type f -exec grep -H "${1:-TrackInitializer}" {} \; ; } 
 
