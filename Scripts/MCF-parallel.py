#!/usr/bin/env python
# coding: utf-8

# core
from functools import partial
import sys, os, re, argparse, platform, subprocess, pickle, cProfile
import numpy as np
from datetime import datetime
import pandas as pd
from scipy import spatial
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

# plotting
import matplotlib.pyplot as plt
# image recognition
import cv2
import imutils
import pytesseract
from pytesseract import image_to_string
import skimage.measure
from skimage import io, data
from skimage.util import img_as_float, img_as_ubyte
from pathlib import Path
# Custom utilities
from utilities import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def the_analysis(window):
    return cv2.GaussianBlur(src=window, ksize=(9, 9), sigmaX=2,)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    set_start_method('spawn')
    start = datetime.now()

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
    parser.add_argument("--sliding-window", type=bool, default=True,
                    action="store", dest="sliding_window",
                    help="Use sliding window approach. Default=True")
    parser.add_argument("--stepsize", type=int,
                    action="store", dest="stepsize",
                    help="Slide step size in pixels (same in x and y directions). Default=500")
    parser.add_argument("--winW", type=int,
                    action="store", dest="winW",
                    help="Window width in pixels. Default=1000")
    parser.add_argument("--winH", type=int,
                    action="store", dest="winH",
                    help="Window height in pixels. Default=1000")
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
    parser.add_argument("--cpus", default=1,
                        action="store", dest="cpus",
                        help="Number of cpus. Will default to 1. Use 'AUTO' to detect logical cores.")

    args = parser.parse_args()

    if args.cpus == "AUTO":
        cpus = get_cpus_avail()
    else:
        cpus = args.cpus

    kwargs = {}
    kwargs['k_blur'] = args.k_blur
    kwargs['C'] = args.C
    kwargs['blocksize'] = args.blocksize
    kwargs['k_laplacian'] = args.k_laplacian
    kwargs['k_dilate'] = args.k_dilate
    kwargs['k_gradient'] = args.k_gradient
    kwargs['k_foreground'] = args.k_foreground

    h, w, c = cv2.imread(args.input).shape
    if not args.winW:
        winW = round((w / 5) / 100) * 100
    else:
        winW = args.winW
    if not args.winH:
        winH = round((h / 5) / 100) * 100
    else:
        winH = args.winH
    if not args.stepsize:
        stepsize = int(min([winH, winW])/2)
    else:
        stepsize = args.stepsize

    if not args.prefix:
        prefix = Path(args.input).stem
    else:
        prefix = args.prefix

    print("[{}] MCF command:\n\t python MCF-parallel.py --prefix {} --sliding-window {} --stepsize {} --winW {} --winH {} --neighborhood {} --dpi {} --debug {} --k_blur {} --C {} --blocksize {} --k_laplacian {} --k_dilate {} --k_gradient {} --k_foreground {} --Amin {} --Amax {:d} --cpus {}".format(
        datetime.now().strftime('%d %b %Y %H:%M:%S'), args.prefix, args.sliding_window, stepsize, winW, winH, args.neighborhood, args.dpi, args.debug, args.k_blur, args.C, args.blocksize, args.k_laplacian, args.k_dilate, args.k_gradient, args.k_foreground, int(args.Amin), int(args.Amax), args.cpus))

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

    suffixes = ["summary.csv", "contour_data.csv", "contour_data.pkl", "tif", "noindex.tif"]
    conflicts = [output_dir_path / "{}.{}".format(prefix, suffix) for suffix in suffixes]

    if any([c.exists() for c in conflicts]):
        print("\tFound existing files of form {}.*".format(output_dir_path / prefix))
        response = query_yes_no("Would you like to DELETE and OVERWRITE?", None)
        if response==True:
            print("\tRemoving and rerunning...")
        else:
            print("\tDelete output files or change output prefix to rerun analysis")
            sys.exit()

    if args.debug:
        print("[{}] Working directory: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), os.getcwd()))
        print("[{}] Output directory: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), output_dir_path))

    image = cv2.imread(str(input_path))
    if not prefix:
        prefix = input_path.stem
    if args.debug:
        print("[{}] Input file: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), input_path.absolute()))

    """Denoise"""
    print("[{}] Denoising image...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    image = cv2.fastNlMeansDenoisingColored(image.copy(), None, 10, 10, 7, 21)
    border_contour='DETECT'
    if border_contour != 'DETECT':
        if args.debug: print("[{}] Border contour given; skipping border detection".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    else:
        print("[{}] Getting image border...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
        border_contour = mcf(image=image, extract_border=True, **kwargs)

    clone = image.copy()
    blank = np.zeros(clone.shape[0:2], dtype=np.uint8)
    border_mask = cv2.drawContours(blank.copy(), border_contour, 0, (255), -1)
    # mask input image (leaves only the area inside the border contour)
    cutout = cv2.bitwise_and(clone, clone, mask=border_mask)

    n_windows = len(list(sliding_window(image=cutout.copy(), stepSize=stepsize, windowSize=(winW, winH))))
    windows = sliding_window(image=cutout.copy(), stepSize=stepsize, windowSize=(winW, winH))

    print("[{}] Finding contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    with Pool(processes=cpus) as pool:
        contours = pool.map(partial(parallel_mcf, **kwargs), windows)
    contours = flatten([c for c in contours if len(c) > 0])
    print("[{}] Size selecting contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    contours = contour_size_selection(contours, Amin=args.Amin, Amax=args.Amax)
    print("[{}] Found {} contours in all windows".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), len(contours)))
    
    print("[{}] Removing redundant contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    contours = remove_redundant_contours(contours)
    print("[{}] Found {} nonredundant contours".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), len(contours)))

    # Export contours
    print("[{}] Exporting contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))   
    if not output_dir_path.is_dir():
        output_dir_path.mkdir(parents=True, exist_ok=True)
    export_contour_data(image=image, contours=contours, conversion_factor=None,
                        units=None, prefix="{}.ALL".format(prefix), output_dir=output_dir_path)

    print("[{}] Contour data exported to".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    print("\t{}".format(Path(output_dir_path) / "{}.summary.csv".format(prefix)))
    print("\t{}".format(Path(output_dir_path) / "{}.contour_data.csv".format(prefix)))
    print("\t{}".format(Path(output_dir_path) / "{}.contour_data.pkl".format(prefix)))

    print("[{}] Plotting...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    render_contour_plots(image=image, border_contour=border_contour, contours=contours, prefix=prefix, dpi=args.dpi, output_dir=output_dir_path)
    print("[{}] Contour plots saved to".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    print("\t{}".format(Path(output_dir_path) / "{}.tif".format(prefix)))
    print("\t{}".format(Path(output_dir_path) / "{}.noindex.tif".format(prefix)))
    print("[{}] Wrote {} nonredundant contours".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), len(contours)))

    end = datetime.now()
    print("[{}] Time elapsed: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), end-start))