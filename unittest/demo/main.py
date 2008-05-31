"""
  main docstring 
"""

print " =====> importing package "
import package

print " =====> importing package.module "
import package.module

print " =====> importing package.subpkg "
import package.subpkg

print " =====> importing package.subpkg.submod "
import package.subpkg.submod

print " =====> invoke package.whereami from main "
package.whereami(globals())

#print " =====> importing package.module_test ... DOES NOT CAUSE nodetests to run the tests in module_test "
#import package.module_test


