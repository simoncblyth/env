#!/usr/bin/python
#
#  http://files.macscripter.net/joy/files/combimgs.py
#
# author: Martin Michel
# created: 07.05.2008
# requires:
# - Mac OS X 10.5
# special thanks to Joe Hewitt!
# <http://www.joehewitt.com/blog/workflows_part_2.php>

# This script combines several images into a new one, placing them directly
# next to each other in horizontal or vertical orientation:
#
# IMGIMG or
#           I
#           M
#           G
#           I
#           M
#           G
#
# The script is called from an AppleScript, which passes several arguments and
# the list of image file paths to be used for the montage. All passed image
# files are assumed to be quadratic and of the same size (all 32x32 or 48x48 etc.),
# as the AppleScript was written to process Mac OS X icons.
#
# Example: If the AppleScript passes 7 image files (size: 32x32) and asks for
# horizontal image orientation, the script will create a new image with the
# following dimensions: width = 7 x 32, height: 32
# The image files are all drawn next to each other with no extra space between
# them.

from AppKit import *
from objc import *
import glob
import sys

app = NSApplication.sharedApplication()

def main():
    """"""
    # [v]ertical or [h]orizontal image orientation?
    acimgorient = sys.argv[1]
    # 'PNG' or 'JPG'
    acimgformat = sys.argv[2]
    # output path
    acimgpath = sys.argv[3]
    # image files to be used for the montage
    imgfiles = sys.argv[4:]
    
    lenimpfiles = len(imgfiles)
    
    # in order to calculate the width and height of the new image, we need
    # to open one sample image and get its size. we assume that all given
    # images are quadratic and of the same size (all images = 32x32 or 48x48, etc.)
    sampleimg = NSImage.alloc().initByReferencingFile_(imgfiles[0].decode('utf-8'))
    sampleimgsize = sampleimg.size()
    quadsize = sampleimgsize.height
    
    # initializing the new image to contain the given import images
    if acimgorient == 'v':
        img = NSImage.alloc().initWithSize_([quadsize, lenimpfiles*quadsize])
    elif acimgorient == 'h':
        img = NSImage.alloc().initWithSize_([lenimpfiles*quadsize,  quadsize])
    
    img.lockFocus()
    
    context = NSGraphicsContext.currentContext()
    context.setImageInterpolation_(NSImageInterpolationHigh)
    
    stackstart = 0
    
    # inserting/drawing the given import images in/to the new image, one
    # directly next to the other
    for imgfile in imgfiles:
        impimg = NSImage.alloc().initByReferencingFile_(imgfile.decode('utf-8'))
        if acimgorient == 'v':
            destRect = ((0, stackstart), (quadsize, quadsize))
        elif acimgorient == 'h':
            destRect = ((stackstart, 0), (quadsize, quadsize))
        srcRect = ((0, 0), (quadsize, quadsize))
        impimg.drawInRect_fromRect_operation_fraction_(destRect, srcRect, NSCompositeSourceOver, 1.0)
        stackstart += quadsize
        
    img.unlockFocus()
    
    # converting the created image to Bitmap representation and saving the result
    # as a JPG or PNG image file.
    tiffData = img.TIFFRepresentation()
    bitmap = NSBitmapImageRep.alloc().initWithData_(tiffData)
    if acimgformat == 'PNG':
        imageData = bitmap.representationUsingType_properties_(NSPNGFileType, None)
    elif acimgformat == 'JPG':
        imageData = bitmap.representationUsingType_properties_(NSJPEGFileType, {NSImageCompressionFactor: 1.0})
    imageData.writeToFile_atomically_(acimgpath.decode('utf-8'), YES)

if __name__ == '__main__':
    main()
