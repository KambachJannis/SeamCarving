# Seam Carving

This is just another implementation of the seam carving technique described in [Seam Carving for Content-Aware Image Resizing](https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj4-seamcarving/imret.pdf). It can remove a specified number of vertical and/or horizontal seams from a single image at a time.

## Requirements

* OpenCV
* NumPy

## Usage

The script needs to be called from the command line with the following arguments:
```
python seamcarving.py -file <IMAGE> -vertical <X> -horizontal <Y>
```
- IMAGE: path to the targeted image file
- X: number of vertical seams to remove
- Y: number of horizontal seams to remove