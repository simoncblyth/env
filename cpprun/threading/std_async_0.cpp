// gcc std_async_0.cpp -std=c++11 -lstdc++ -o /tmp/std_async_0 && /tmp/std_async_0

// https://thispointer.com/c11-multithreading-part-9-stdasync-tutorial-example/

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <future>

using namespace std::chrono;

std::string fetchDataFromDB(std::string recvdData)
{
    std::cout << "[fetchDataFromDB " << std::endl ; 
    std::this_thread::sleep_for(seconds(5));
    std::cout << "]fetchDataFromDB " << std::endl ; 
    return "DB_" + recvdData;
}
std::string fetchDataFromFile(std::string recvdData)
{
    std::cout << "[fetchDataFromFile " << std::endl ; 
    std::this_thread::sleep_for(seconds(5));
    std::cout << "]fetchDataFromFile " << std::endl ; 
    return "File_" + recvdData;
}
int main()
{
    system_clock::time_point start = system_clock::now();

    // normal sequential running of two funcs
    // std::string dbData = fetchDataFromDB("Data");
    
    std::future<std::string> future_dbData  = std::async(std::launch::async, fetchDataFromDB, "Data");

    std::string fileData = fetchDataFromFile("Data");

    // this blocks until the result arrives
    std::string dbData = future_dbData.get();

    auto end = system_clock::now();
    auto diff = duration_cast < std::chrono::seconds > (end - start).count();
    std::cout << "Total Time Taken = " << diff << " Seconds" << std::endl;
    std::string data = dbData + " :: " + fileData;
    std::cout << "Data = " << data << std::endl;
    return 0;
}
