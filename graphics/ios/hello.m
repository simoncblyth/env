#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface MyDelegate : UIResponder< UIApplicationDelegate >
@end

@implementation MyDelegate
- ( BOOL ) application: ( UIApplication * ) application
           didFinishLaunchingWithOptions: ( NSDictionary * ) launchOptions {
  UIWindow *window = [ [ [ UIWindow alloc ] initWithFrame: 
    [ [ UIScreen mainScreen ] bounds ] ] autorelease ];
  window.backgroundColor = [ UIColor whiteColor ];
  UILabel *label = [ [ UILabel alloc ] init ];
  label.text = @"Hello, World!";
  label.center = CGPointMake( 100, 100 );
  [ label sizeToFit ];
  [ window addSubview: label ];
  [ window makeKeyAndVisible ];
  [ label release ];

  return YES;
}
@end

int main( int argc, char *argv[ ] )
{
  UIApplicationMain( argc, argv, nil, NSStringFromClass( [ MyDelegate class ] ) );
}
