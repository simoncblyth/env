/*
  cmak-
  cmak-cc timecheck.cc
  cmak-bin timecheck
  /c/usr/local/env/tools/cmak/build/Debug/timecheck.exe


*/


// http://stackoverflow.com/questions/7685762/windows-7-timing-functions-how-to-use-getsystemtimeadjustment-correctly

#include <windows.h>
#include <iostream>
#include <iomanip>

int main()
{
    FILETIME fileStart;
    GetSystemTimeAsFileTime(&fileStart);
    ULARGE_INTEGER start;
    start.HighPart = fileStart.dwHighDateTime;
    start.LowPart = fileStart.dwLowDateTime;

    for (int i=20; i>0; --i)
    {
        FILETIME timeStamp1;
        ULARGE_INTEGER ts1;

        GetSystemTimeAsFileTime(&timeStamp1);

        ts1.HighPart = timeStamp1.dwHighDateTime;
        ts1.LowPart  = timeStamp1.dwLowDateTime;

        std::cout << "Timestamp: " << std::setprecision(20) << (double)(ts1.QuadPart - start.QuadPart) / 10000000 << std::endl;

    }

    DWORD dwTimeAdjustment = 0, dwTimeIncrement = 0, dwClockTick;
    BOOL fAdjustmentDisabled = TRUE;
    GetSystemTimeAdjustment(&dwTimeAdjustment, &dwTimeIncrement, &fAdjustmentDisabled);

    std::cout << "\nTime Adjustment disabled: " << fAdjustmentDisabled
        << "\nTime Adjustment: " << (double)dwTimeAdjustment/10000000
        << "\nTime Increment: " << (double)dwTimeIncrement/10000000 << std::endl;

}
