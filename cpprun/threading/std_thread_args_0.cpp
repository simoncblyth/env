// gcc std_thread_args_0.cpp -std=c++11 -lstdc++ -o /tmp/std_thread_args_0 && /tmp/std_thread_args_0 

// https://eli.thegreenplace.net/2016/the-promises-and-challenges-of-stdasync-task-based-parallelism-in-c11/

#include <thread>
#include <iostream>
#include <vector> 
#include <numeric>   // std::accumulate


void accumulate_block_worker(int* data, size_t count, int* result) {
  *result = std::accumulate(data, data + count, 0);
}

/**
The result is communicated back to the caller via a pointer argument, since a std::thread cannot have a return value. 
**/

void use_worker_in_std_thread() {
  std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8};
  int result;
  std::thread worker(accumulate_block_worker,
                     v.data(), v.size(), &result);
  worker.join();
  std::cout << "use_worker_in_std_thread computed " << result << "\n";
}




#include <future>

int accumulate_block_worker_ret(int* data, size_t count) {
  return std::accumulate(data, data + count, 0);
}

void use_worker_in_std_async() {
  std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8};
  std::future<int> fut = std::async(std::launch::async, accumulate_block_worker_ret, v.data(), v.size());
  std::cout << "use_worker_in_std_async computed " << fut.get() << "\n";
}


int main(int argc, char** argv)
{
    //use_worker_in_std_thread() ;  
    use_worker_in_std_async() ;  

    return 0 ; 
}
