# Build an iPhone app from the command line
# http://commandlinefanatic.com/cgi-bin/showarticle.cgi?article=art024
#
# Running doesnt work, get Safari cannot download 
#

XCODE_BASE=/Applications/Xcode.app/Contents
SIMULATOR_BASE=$(XCODE_BASE)/Developer/Platforms/iPhoneSimulator.platform
IPHONESIMULATOR=iPhoneSimulator7.1.sdk
SIMULATOR=$(SIMULATOR_BASE)/Developer/Applications/iPhone\ Simulator.app/Contents/MacOS/iPhone\ Simulator
FRAMEWORKS=$(SIMULATOR_BASE)/Developer/SDKs/$(IPHONESIMULATOR)/System/Library/Frameworks/
INCLUDES=$(SIMULATOR_BASE)/Developer/SDKs/$(IPHONESIMULATOR)/usr/include
ISYSROOT=$(XCODE_BASE)/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/$(IPHONESIMULATOR)

	
hello: hello.m
	  clang -arch i386 \
    -mios-simulator-version-min=7.1 \
    -fobjc-abi-version=2 \
    -isysroot $(ISYSROOT) \
    -o $@ $< -framework Foundation -framework UIKit


.PHONY: run


run:
	$(SIMULATOR) -SimulateApplication ./hello
