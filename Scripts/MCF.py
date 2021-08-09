#!/usr/bin/env python
# coding: utf-8

# core
import sys, argparse
from datetime import datetime
# image recognition
import cv2
from pathlib import Path
# Custom utilities
from utilities import *
from imagetools import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    print("""
   __  ___   _          _               
  /  |/  /  (_)  ___   (_)              
 / /|_/ /  / /  / _ \ / /               
/_/__/_/  /_/  /_//_//_/                
 / ___/ ___   ___  / /_ ___  __ __  ____
/ /__  / _ \ / _ \/ __// _ \/ // / / __/
\___/__\___//_//_/\__/ \___/\_,_/ /_/   
  / __/  (_)  ___  ___/ / ___   ____    
 / _/   / /  / _ \/ _  / / -_) / __/    
/_/    /_/  /_//_/\_,_/  \__/ /_/       
                                        
""")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                    help="Filepath to query image")
    parser.add_argument("--output_dir", type=str, default='./',
                    help="Path to output directory. Default='./'")
    parser.add_argument("--prefix", type=str,
                    action="store", dest="prefix",
                    help="New prefix for output files. By default the new files will reflect the input file's basename")
    # parser.add_argument("--stepsize", type=int,
    #                 action="store", dest="stepsize",
    #                 help="Slide step size in pixels (same in x and y directions). Default=500")
    # parser.add_argument("--winW", type=int,
    #                 action="store", dest="winW",
    #                 help="Window width in pixels. Default=1000")
    # parser.add_argument("--winH", type=int,
    #                 action="store", dest="winH",
    #                 help="Window height in pixels. Default=1000")
    parser.add_argument("--neighborhood", type=int, default=10,
                    action="store", dest="neighborhood",
                    help="Neighborhood size in pixels determining a unique contour. Default=10")
    parser.add_argument("--dpi", type=int, default=300,
                    action="store", dest="dpi",
                    help="Output image resolution in pixels. Default=300")
    parser.add_argument("--debug", type=bool, default=False,
                    action="store", dest="debug",
                    help="Writes debugging information and plots more steps")
    parser.add_argument("--k_blur", type=int, default=9,
                    action="store", dest="k_blur",
                    help="blur kernel size; must be odd. Default=9")
    parser.add_argument("--C", type=int, default=3,
                    action="store", dest="C",
                    help="Constant subtracted from mean during adaptive Gaussian smoothing. Default=3")
    parser.add_argument("--blocksize", type=int, default=15,
                    action="store", dest="blocksize",
                    help="Neighborhood size for calculating adaptive Gaussian threshold; must be odd. Default=15")
    parser.add_argument("--k_laplacian", type=int, default=5,
                    action="store", dest="k_laplacian",
                    help="Laplacian kernel size; must be odd. Default=5")
    parser.add_argument("--k_dilate", type=int, default=5,
                    action="store", dest="k_dilate",
                    help="Dilation kernel size; must be odd. Default=5")
    parser.add_argument("--k_gradient", type=int, default=5,
                    action="store", dest="k_gradient",
                    help="Gradient kernel size; must be odd. Default=3")
    parser.add_argument("--k_foreground", type=int, default=7,
                    action="store", dest="k_foreground",
                    help="Foregound clean up kernel size; must be odd. Default=7")
    parser.add_argument("--Amin", type=int, default=50,
                        action="store", dest="Amin",
                        help="Minimum contour area in pixel")
    parser.add_argument("--Amax", type=int, default=10e6,
                        action="store", dest="Amax",
                        help="Maximum contour area in pixels")

    args = parser.parse_args()

    kwargs = {}
    kwargs['k_blur'] = args.k_blur
    kwargs['C'] = args.C
    kwargs['blocksize'] = args.blocksize
    kwargs['k_laplacian'] = args.k_laplacian
    kwargs['k_dilate'] = args.k_dilate
    kwargs['k_gradient'] = args.k_gradient
    kwargs['k_foreground'] = args.k_foreground

    if not args.prefix:
        prefix = Path(args.input).stem
    else:
        prefix = args.prefix


    print("[{}] MCF command:\n\t python MCF.py --prefix {} --neighborhood {} --dpi {} --debug {} --k_blur {} --C {} --blocksize {} --k_laplacian {} --k_dilate {} --k_gradient {} --k_foreground {} --Amin {} --Amax {:d}".format(
        datetime.now().strftime('%d %b %Y %H:%M:%S'), args.prefix, args.neighborhood, args.dpi, args.debug, args.k_blur, args.C, args.blocksize, args.k_laplacian, args.k_dilate, args.k_gradient, args.k_foreground, int(args.Amin), int(args.Amax)))

    if args.input.startswith("~/"):
        input_path = Path(Path.home() / args.input[2:])
    else:
        input_path = Path(args.input).absolute().resolve()
    if args.output_dir.startswith("~/"):
        output_dir_path = Path(Path.home() / args.output_dir[2:])
    else:
        output_dir_path = Path(args.output_dir).absolute().resolve()

    print("[{}] Input file: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), input_path.absolute()))
    print("[{}] Output directory: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), output_dir_path.absolute()))

    suffixes = ["contour_data.csv", "contour_data.json", "tif", "noindex.tif"]
    conflicts = [output_dir_path / "{}.{}".format(prefix, suffix) for suffix in suffixes]

    if any([c.exists() for c in conflicts]):
        print("\tFound existing files of form {}.*".format(output_dir_path / prefix))
        response = query_yes_no("Would you like to DELETE and OVERWRITE?", None)
        if response==True:
            print("\tRemoving and rerunning...")
        else:
            print("\tDelete output files or change output prefix to rerun analysis")
            sys.exit()

    process_image(image_path=input_path, prefix=prefix, Amin=args.Amin, Amax=args.Amax, neighborhood=args.neighborhood, output_dir=output_dir_path, dpi=args.dpi, debug=args.debug, **kwargs)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print("[{}] Time elapsed: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), end-start))