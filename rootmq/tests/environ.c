/*

    Was a little surprised to see that the SCons/SCT : CommandOutputBuilder 
    appends the $TEST_DIR to PATH and LD_LIBRARY_PATH 
    in the cloned test running env
    
        env.AppendENVPath('PATH', cwdir)
        env.AppendENVPath('LD_LIBRARY_PATH', cwdir)

*/

#include <unistd.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    printf("starting... %s\n", argv[0]);
    extern char **environ;
    int e = 0;
    while (environ[e] != NULL) {
    	printf("%s\r\n", environ[e]);
    	e++;
    }
    return(0);
}





