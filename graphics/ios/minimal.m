/*
http://commandlinefanatic.com/cgi-bin/showarticle.cgi?article=art024

delta:ios blyth$ clang -o minimal minimal.m -framework Foundation 
delta:ios blyth$ ./minimal 
2015-03-31 11:53:38.739 minimal[60320:507] abc123

*/

#import <Foundation/NSString.h>

int main( int argc, char *argv[ ] )
{
    NSString *s1 = @"abc";
    NSString *s2 = @"123";

    NSLog( @"%@", [ s1 stringByAppendingString: s2 ] );
}
