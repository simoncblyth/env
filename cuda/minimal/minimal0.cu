// nvcc -arch=sm_30 minimal0.cu -run ; rm a.out
//  
// nvcc -o /tmp/minimal0 minimal0.cu

// note no cuda includes, somehow nvcc automates that ?

#include <stdio.h>
#define N 10


__global__ void fadd(float *a, float *b) 
{
    int i = blockIdx.x;
    if (i<N) b[i] = float(2)*a[i];

    printf("fadd.. %d \n", i );

}
__global__ void dadd(double *a, double *b) 
{
    int i = blockIdx.x;
    if (i<N) b[i] = double(2)*a[i];

    printf("dadd.. %d \n", i );

}
__global__ void iadd(int *a, int *b) 
{
    int i = blockIdx.x;
    if (i<N) b[i] = int(2)*a[i];

    printf("iadd.. %d \n", i );

}

/*
// avoid problems with clang and templates in device code

template void add<int>(int*, int*) ; 
template void add<float>(float*, float*) ; 
template void add<double>(double*, double*) ; 

template <typename T>
*/


template <typename T>
struct Minimal
{
    T ha[N]; 
    T hb[N]; 
    T* da ; 
    T* db ; 

    Minimal()
    {
        init();
    }

    void init()
    {
        cudaMalloc((void **)&da, N*sizeof(T));
        cudaMalloc((void **)&db, N*sizeof(T));
        for (int i = 0; i<N; ++i) ha[i] = T(i);
    }

    void copyto()
    {
        cudaMemcpy(da, ha, N*sizeof(T), cudaMemcpyHostToDevice);
    }
    void copyback()
    {
        cudaMemcpy(hb, db, N*sizeof(T), cudaMemcpyDeviceToHost);
    }


    void launch();
    void dump();

    void cleanup()
    {
        cudaFree(da);
        cudaFree(db);
    }
};



template<>
void Minimal<float>::launch()
{
    printf("Minimal<float>::launch \n");
    fadd<<<N, 1>>>(da, db);
}

template<>
void Minimal<int>::launch()
{
    printf("Minimal<int>::launch \n");
    iadd<<<N, 1>>>(da, db);
}

template<>
void Minimal<double>::launch()
{
    printf("Minimal<double>::launch \n");
    dadd<<<N, 1>>>(da, db);
}




template<>
void Minimal<float>::dump()
{
    printf("Minimal<float>::dump \n");
    for (int i = 0; i<N; ++i) printf("%f\n", hb[i]);
}

template<>
void Minimal<int>::dump()
{
    printf("Minimal<int>::dump \n");
    for (int i = 0; i<N; ++i) printf("%d\n", hb[i]);
}

template<>
void Minimal<double>::dump()
{
    printf("Minimal<double>::dump \n");
    for (int i = 0; i<N; ++i) printf("%e\n", hb[i]);
}






int main(int /*argc*/, char** /*argv*/) 
{

    //Minimal<int> m ; 
    Minimal<double> m ; 
    //Minimal<float> m ; 

    m.copyto();
    m.launch();
    m.copyback();
    m.dump();
    m.cleanup();

    return 0;
}
