//  gcc std_future_return_large_vector.cpp -std=c++11 -lstdc++ -o /tmp/std_future_return_large_vector && /tmp/std_future_return_large_vector

// https://stackoverflow.com/questions/31613691/how-to-properly-return-large-data-from-a-stdfuture-in-c11


#include <unistd.h>

#include <iostream>
#include <chrono>

#include <future>
#include <vector>

using timepoint = std::chrono::time_point<std::chrono::system_clock>;

timepoint start_return;

std::vector< int > test_return_of_large_vector(void)
{
    std::vector< int > ret(100000000);
    start_return = std::chrono::system_clock::now();
    return ret;                          // MOVE1
}


int main(void)
{

    timepoint start = std::chrono::system_clock::now();
    auto ret = test_return_of_large_vector();
    timepoint end = std::chrono::system_clock::now();

    auto dur_create_and_return = std::chrono::duration_cast< std::chrono::milliseconds >(end - start);
    auto dur_return_only = std::chrono::duration_cast< std::chrono::milliseconds >(end - start_return);

    std::cout << "create & return time : " << dur_create_and_return.count() << "ms\n";
    std::cout << "return time : " << dur_return_only.count() << "ms\n";

    auto future = std::async(std::launch::async, test_return_of_large_vector);
    sleep(3); // wait long enough for the future to finish its work

    start = std::chrono::system_clock::now();
    ret = future.get();                  // MOVE2
    end = std::chrono::system_clock::now();

    // mind that the roles of start and start_return have changed
    dur_return_only = std::chrono::duration_cast< std::chrono::milliseconds >(end - start);
    dur_create_and_return = std::chrono::duration_cast< std::chrono::milliseconds >(end - start_return);

    std::cout << "duration since future finished: " << dur_create_and_return.count() << "ms\n";
    std::cout << "return time from future: " << dur_return_only.count() << "ms\n";

    return 0;
}
