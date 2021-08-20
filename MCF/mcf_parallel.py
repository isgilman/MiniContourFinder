#!/usr/bin/env python
# coding: utf-8

# core
from functools import partial
import sys, os, argparse
import numpy as np
from datetime import datetime
from multiprocessing import Pool, set_start_method
# image recognition
import cv2
from pathlib import Path
# Custom utilities
try:
    from helpers import *
    from imagetools import *
except ModuleNotFoundError:
    from MCF.helpers import *
    from MCF.imagetools import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    parser.add_argument("-i", "--input", metavar='', type=str,
                    help="filepath to query image")
    parser.add_argument("-o", "--output_dir", metavar='', type=str, 
                    default='./', help="path to output directory. Default='./'")
    parser.add_argument("-S", "--stepsize", metavar='',
                    type=int, action="store", dest="stepsize",
                    help="Slide step size in pixels (same in x and y directions). Default=500")
    parser.add_argument("-W", "--winW", metavar='', type=int,
                    action="store", dest="winW",
                    help="Window width in pixels. Default=1000")
    parser.add_argument("-H", "--winH", metavar='',
                    type=int, action="store", dest="winH",
                    help="Window height in pixels. Default=1000")
    parser.add_argument("--cpus", default=1,
                        action="store", dest="cpus",
                        help="Number of cpus. Will default to 1. Use 'AUTO' to detect logical cores.")
    parser.add_argument("-p", "--prefix", metavar='', type=str,
                    action="store", dest="prefix",
                    help="new prefix for output files. By default the new files will reflect the input file's basename")
    parser.add_argument("-D", "--detectScaleBar", metavar='', type=bool,    
                    default=False, action="store", dest="detectScaleBar",
                    help="automated scale bar detection.")
    parser.add_argument("-d", "--dpi", metavar='', type=int, default=300,
                    action="store", dest="dpi",
                    help="Output image resolution in pixels. Default=300")
    parser.add_argument("--debug", type=bool, default=False,
                    action="store", dest="debug",
                    help="writes debugging information and plots more steps")
    parser.add_argument("-n", "--neighborhood", metavar='', type=int,
                    default=10, action="store", dest="neighborhood",
                    help="neighborhood size in pixels determining a unique contour. Default=10")
    parser.add_argument("-kb", "--k_blur", metavar='', type=int, 
                    default=9, action="store", dest="k_blur",
                    help="blur kernel size; must be odd. Default=9")
    parser.add_argument("-c", "--C", metavar='', type=int, default=3,
                    action="store", dest="C",
                    help="constant subtracted from mean during adaptive Gaussian smoothing. Default=3")
    parser.add_argument("-B", "--blocksize", metavar='', type=int, 
                    default=15, action="store", dest="blocksize",
                    help="neighborhood size for calculating adaptive Gaussian threshold; must be odd. Default=15")
    parser.add_argument("-kl", "--k_laplacian", metavar='', type=int, 
                    default=5, action="store", dest="k_laplacian",
                    help="Laplacian kernel size; must be odd. Default=5")
    parser.add_argument("-kd", "--k_dilate", metavar='', type=int, 
                    default=5, action="store", dest="k_dilate",
                    help="dilation kernel size; must be odd. Default=5")
    parser.add_argument("-kg", "--k_gradient", metavar='', type=int, 
                    default=5, action="store", dest="k_gradient",
                    help="gradient kernel size; must be odd. Default=3")
    parser.add_argument("-kf", "--k_foreground", metavar='', type=int, 
                    default=7, action="store", dest="k_foreground",
                    help="Foregound clean up kernel size; must be odd. Default=7")
    parser.add_argument("-a", "--Amin", metavar='', type=int, 
                        default=50, action="store", dest="Amin",
                        help="Minimum contour area in pixel")
    parser.add_argument("-A", "--Amax", metavar='', type=int, 
                        default=10e6, action="store", dest="Amax",
                        help="Maximum contour area in pixels")

    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit()

    kwargs = {}
    kwargs['k_blur'] = args.k_blur
    kwargs['C'] = args.C
    kwargs['blocksize'] = args.blocksize
    kwargs['k_laplacian'] = args.k_laplacian
    kwargs['k_dilate'] = args.k_dilate
    kwargs['k_gradient'] = args.k_gradient
    kwargs['k_foreground'] = args.k_foreground

    h, w, _ = cv2.imread(args.input).shape
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

    print("[{}] command:\n\t python mcf_parallel.py --prefix {} --stepsize {} --winW {} --winH {} --neighborhood {} --dpi {} --debug {} --k_blur {} --C {} --blocksize {} --k_laplacian {} --k_dilate {} --k_gradient {} --k_foreground {} --Amin {} --Amax {:d} --cpus {}".format(
        datetime.now().strftime('%d %b %Y %H:%M:%S'), args.prefix, stepsize, winW, winH, args.neighborhood, args.dpi, args.debug, args.k_blur, args.C, args.blocksize, args.k_laplacian, args.k_dilate, args.k_gradient, args.k_foreground, int(args.Amin), int(args.Amax), args.cpus))
    if args.cpus == "AUTO":
        cpus = get_cpus_avail()
    else:
        cpus = args.cpus
    print("[{}] Using {} cpus".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), cpus))

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

    suffixes = ["contour_data.csv", "contour_data.json", "pdf", "noindex.pdf"]
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

    image = cv2.imread(input_path.as_posix())
    if image == None:
        image = vector2cv2(image_path.as_posix())

    if not prefix:
        prefix = input_path.stem
    if debug:
        print("[{}] Input file: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), input_path.absolute()))

    """Denoise"""
    wdir = image_path.parent
    results = wdir.glob("{}.denoise.*".format(image_path.stem))
    try:
        denoised_path = next(results)
        denoise = cv2.imread(denoised_path.as_posix())
        print("[{}] Found existing denoised file ({})".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), denoised_path.as_posix()))
    except StopIteration:
        print("[{}] Denoising image...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
        denoise = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        denoised_path = Path("{}/{}.denoise{}".format(wdir.as_posix(), image_path.stem, ".png"))
        cv2.imwrite(filename=denoised_path.as_posix(), img=denoise)
        print("[{}] Created temporary denoised file ({})".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), denoised_path.as_posix()))

    print("[{}] Getting image border...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    border_contour = mcf(image=denoise, extract_border=True, **kwargs)

    # mask input image (leaves only the area inside the border contour)
    clone = denoise.copy()
    blank = np.zeros(clone.shape[0:2], dtype=np.uint8)
    border_mask = cv2.drawContours(blank.copy(), border_contour, 0, (255), -1)
    cutout = cv2.bitwise_and(clone, clone, mask=border_mask)

    # Break image into windows
    windows = list(sliding_window(image=cutout.copy(), stepSize=stepsize, windowSize=(winW, winH)))
    windows = [w for w in windows if w[2].sum() > 0]
    print("[{}] Finding contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    with Pool(processes=cpus) as pool:
        contours = pool.map(partial(parallel_mcf, **kwargs), windows)
    contours = flatten([c for c in contours if len(c) > 0])
    print("[{}] Size selecting contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    contours = contour_size_selection(contours, Amin=args.Amin, Amax=args.Amax)
    print("[{}] Found {} contours in all windows".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), len(contours)))
    
    print("[{}] Removing redundant contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    contours = remove_redundant_contours(contours, neighborhood=args.neighborhood)
    print("[{}] Found {} nonredundant contours".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), len(contours)))

    # Export contours
    print("[{}] Exporting contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))   
    if not output_dir_path.is_dir():
        output_dir_path.mkdir(parents=True, exist_ok=True)
    export_contour_data(image=image, contours=contours, conversion_factor=None,
                        units=None, prefix="{}.ALL".format(prefix), output_dir=output_dir_path)

    print("[{}] Plotting...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    render_contour_plots(image=image, border_contour=border_contour, contours=contours, prefix=prefix, dpi=args.dpi, output_dir=output_dir_path)

    end = datetime.now()
    print("[{}] Time elapsed: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), end-start))