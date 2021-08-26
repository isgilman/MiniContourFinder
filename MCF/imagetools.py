#!/usr/bin/env python
# coding: utf-8

# core
import sys, os, json, codecs, uuid
import numpy as np
from datetime import datetime
import pandas as pd
from scipy import spatial
from tqdm import tqdm
from pathlib import Path, PosixPath
from typing import Union
# plotting
import matplotlib
if sys.platform == 'darwin':
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# image related
import cv2
from PIL import Image
from pytesseract import image_to_string
from PyQt5.QtGui import QPixmap, QImage

# Custom utils
try:
    from helpers import *
except ModuleNotFoundError:
    from MCF.helpers import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def flood_fill(image, progress_bar=True):
    """Flood fill from the edges."""
    height, width = image.shape[:2]
    for row in range(height):
        if image[row, 0] == 255:
            cv2.floodFill(image, None, (0, row), 0)
        if image[row, width-1] == 255:
            cv2.floodFill(image, None, (width-1, row), 0)
    if progress_bar:
        for col in tqdm(range(width), desc='Flooding background', leave=False):
            if image[0, col] == 255:
                cv2.floodFill(image, None, (col, 0), 0)
            if image[height-1, col] == 255:
                cv2.floodFill(image, None, (col, height-1), 0)
    else:
        for col in range(width):
            if image[0, col] == 255:
                cv2.floodFill(image, None, (col, 0), 0)
            if image[height-1, col] == 255:
                cv2.floodFill(image, None, (col, height-1), 0)

    return image

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mcf(image, k_blur=9, C=3, blocksize=15, k_laplacian=5, k_dilate=5, k_gradient=3, k_foreground=7,
        extract_border=False, offsetX=0, offsetY=0, skip_flood=False, debug=False, progress_bar=True):
    """Extracts contours from an image. Can be used to extract a contour surrounding the
    entire foreground border.

    Parameters
    ----------
    image : <numpy.ndarray> Query image
    k_blur : <int> 9; blur kernel size; must be odd
    C : <int> 3; constant subtracted from mean during adaptive Gaussian smoothing
    blocksize : <int> 15; neighborhood size for calculating adaptive Gaussian threshold; must be odd
    k_laplacian : <int> 5; laplacian kernel size; must be odd
    k_dilate : <int> 5; dilation kernel size; must be odd
    k_gradient : <int> 3; gradient kernel size; must be odd
    k_foreground : <int> 7; foregound clean up kernel size; must be odd
    extract_border : <bool> False; extract background o
    debug : <bool> writes debugging information and plots each step

    Returns
    -------
    contours : <list> A list of contours"""
    """Gray"""
    if debug: print("[PID {}] Gray...".format(os.getpid()))
    gray = cv2.cvtColor(src = image, code=cv2.COLOR_RGB2GRAY)
    """Adaptive histogram normalization"""
    gridsize = int(min(0.01*max(image.shape[:2]), 8))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    hequalized = clahe.apply(gray)
    """Blur"""
    if debug: print("[PID {}] Blur...".format(os.getpid()))
    blur = cv2.GaussianBlur(src=hequalized, ksize=(k_blur, k_blur), sigmaX=2, )
    """Adaptive Gaussian threshold"""
    if debug: print("Adapt Gauss Thresh...")
    thresh = cv2.adaptiveThreshold(src=blur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=blocksize, C=C)
    """Laplacian"""
    if debug: print("Laplacian...")
    laplacian = cv2.Laplacian(src=thresh, ddepth=cv2.CV_16S, ksize=k_laplacian, )
    """Dilate"""
    if debug: print("Dilate...")
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(k_dilate, k_dilate))
    dilate = cv2.dilate(laplacian, kernel=kernel, iterations=1)
    """Morphological gradient"""
    if debug: print("Gradient...")
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(k_gradient, k_gradient))
    gradient = cv2.morphologyEx(dilate, cv2.MORPH_GRADIENT, kernel=kernel, iterations=1)
    """Binarize"""
    if debug: print("Binarize...")
    tozero = cv2.threshold(gradient, 127, 255, cv2.THRESH_TOZERO)
    tozero = np.uint8(np.uint8(tozero[1]))
    binary = cv2.inRange(tozero, 0, 100)
    """Foreground clean up"""
    if debug: print("Foreground...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_foreground, k_foreground))
    foreground = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)
    """Flood from outside"""
    if debug: print("Flood fill...")
    if skip_flood:
        flood = foreground.copy()
    else:
        flood = flood_fill(foreground, progress_bar=progress_bar)
    if extract_border:
        """Get border"""
        if debug: print("Getting border...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        background = cv2.dilate(flood, kernel, iterations=2)
        background_contours, _ = cv2.findContours(image=background.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        border_contour = [max(background_contours, key=cv2.contourArea)]
        return border_contour

    """Find contours"""
    if debug: print("Drawing contours...")
    contours, _ = cv2.findContours(image=flood.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    return [c + [offsetX, offsetY] for c in contours]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parallel_mcf(window, **kwargs):
    return mcf(image=window[2], offsetX=window[0], offsetY=window[1], progress_bar=False, **kwargs)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def contour_size_selection(contours, Amin=50, Amax=10e6):
    """Selections contours based on perimeter and area

    Parameters
    ----------
    contours : <list> A list of contours
    Amin : <int> 50; Minimum area in pixels
    Amax : <int> 10e6; Maximum area in pixels

    Returns
    -------
    contours : <list> A list of contours
    """

    return [c for c in contours if Amax >= cv2.contourArea(c) >= Amin]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def smooth_contours(contours, epsilon=1):
    """Returns slightly smoothed and convex hulls versions of the input contours.

    Parameters
    ----------
    contours : <list> A list of contours

    Returns
    -------
    smoothed : <list> A list of smoothed contours
    hulls : <list> A list of contour convex hulls
    """
    smoothed = []
    hulls = []
    for c in contours:
        smoothed.append(cv2.approxPolyDP(curve=c, epsilon=epsilon, closed=True))
        hulls.append(cv2.convexHull(c))

    return smoothed, hulls

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sliding_window(image, stepSize, windowSize):
    """Creates a generator of overlapping windows that tile across an image.

    Parameters
    ----------
    image : <numpy.ndarray> Query image
    stepsize : <int> Slide step size in pixels (currently the same in x and y directions)
    windowSize: <int>

    Returns
    -------
    smoothed : <list> A list of smoothed contours
    hulls : <list> A list of contour convex hulls
    """
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sliding_contour_finder(image, stepsize, winW, winH, border_contour, skip_flood=False, debug=False, **kwargs):
    """Uses a sliding-window approach to find contours across a large image. Uses KDTree algorithm to
    remove duplicated contours from overlapping windows.

    Parameters
    ----------
    image : <numpy.ndarray> Query image
    stepsize : <int> Slide step size in pixels (currently the same in x and y directions)
    winW : <int> Window width in pixels
    winH : <int> Window height in pixels
    **kwargs : Kwargs passed to `mcf`

    Returns
    -------
    contours : <list> A list of contours
    smooth_contours : <list> A list of smoothed contours"""

    """Create windows for mini contour finder"""
    if debug: print("Creating windows...")

    # Create image of border
    clone = image.copy()
    blank = np.zeros(clone.shape[0:2], dtype=np.uint8)
    border_mask = cv2.drawContours(blank.copy(), border_contour, 0, (255), -1)
    # mask input image (leaves only the area inside the border contour)
    cutout = cv2.bitwise_and(clone, clone, mask=border_mask)

    n_windows = len(list(sliding_window(image=cutout.copy(), stepSize=stepsize, windowSize=(winW, winH))))
    windows = sliding_window(image=cutout.copy(), stepSize=stepsize, windowSize=(winW, winH))

    contours = []
    for i, (x,y,window) in tqdm(enumerate(windows), total=n_windows, desc='Windows'):
        if debug: print(("Window {}, x0: {}, y0: {}, shape: {}".format(i,x,y,np.shape(window))))
        if window.shape[0] != winH or window.shape[1] != winW: continue
        if window.sum() == 0: continue
        """Running mini contour finder in window"""
        if debug: print("Running mini contour finder...")
        contours += mcf(window, skip_flood=skip_flood, progress_bar=False, offsetX=x, offsetY=y, **kwargs)

    return contours

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def remove_redundant_contours(contours, neighborhood=10):
    moments = []
    nonredundant = []
    for c in contours:
        cX,cY = contour_xy(c)

        if len(moments)==0:
            nonredundant.append(c)
            moments.append([cX, cY])
        else: # if previous moments exist, find the distance and index of the nearest neighbor
            distance, _ = spatial.KDTree(moments).query([cX, cY])
            if distance > neighborhood: # add point if moment falls outside of neighborhood
                nonredundant.append(c)
                moments.append([cX, cY])

    return nonredundant

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def getContourRBG(contour, image):
    clone = image.copy()
    mask = np.zeros_like(clone)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contour, 0, 255, -1)  # Draw filled contour in mask
    out = np.zeros_like(image)  # Extract out the object and place into output image
    out[mask == 255] = image[mask == 255]
    pixelpoints = np.transpose(np.nonzero(mask))

    return image[flatten(pixelpoints[:, :2])[::2], flatten(pixelpoints[:, :2])[1::2]].mean(axis=0) / 255

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def contour_xy(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return (int((M["m10"] / M["m00"]))-10, int((M["m01"] / M["m00"])))
    else: 
        return (0,0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def export_contour_data(contourDF:pd.DataFrame, prefix:str, image:np.ndarray, conversion_factor=None, units=None, output_dir="./", get_color=False):
    """Exports contours, contour metadata, and summary statistics.

    Parameters
    ----------
    image : <np.ndarray> Image
    contourDF : <np.ndarray> Contours to export
    conversion_factor : <float> Conversion factort to convert pixel length. Default=None
    units : <str> Associated conversion factor units. Default=None
    prefix : <str> Prefix for output files (e.g. PREFIX.contour_data.pkl, PREFIX.contour_data.csv)
    output_dir : <str> Path to output directory. Default='./' """
    
    contourDF["area_pixels"] = np.array(list(map(cv2.contourArea, contourDF["contour"].values)), dtype=object)
    contourDF["moment_XY"] = [contour_xy(c) for c in contourDF["contour"].values]
    cRXY = np.array(list(map(cv2.minEnclosingCircle, contourDF["contour"].values)), dtype=object)
    contourDF["min_circle_xy"] = cRXY[:,0]
    contourDF["min_circle_r"] = cRXY[:,1]
    bbox = np.array([cv2.boundingRect(c) for c in contourDF["contour"].values])
    contourDF["bbox_x"], contourDF["bbox_y"], contourDF["bbox_h"], contourDF["bbox_w"] = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]
    contourDF["bbox_area"] = contourDF["bbox_w"]*contourDF["bbox_h"]
    contourDF["aspect_ratio"] = contourDF["bbox_w"]/contourDF["bbox_h"]
    contourDF["convex_hull"] = [cv2.convexHull(c) for c in contourDF["contour"].values]
    contourDF["convexity"] = [cv2.isContourConvex(c) for c in contourDF["contour"].values]
    contourDF["solidity"] = [float(cv2.contourArea(c))/cv2.contourArea(cv2.convexHull(c)) for c in contourDF["contour"].values]
    contourDF["equivalent_diameter"] = [np.sqrt(4*cv2.contourArea(c)/np.pi) for c in contourDF["contour"].values]

    if conversion_factor and units:
        contourDF["area_{}^2".format(units)] = contourDF["area_pixels"]*(conversion_factor**2)

    if get_color:
        if type(image)==str:
            image = cv2.imread(image)
        contourDF["RBG"] = np.array(list(map(getContourRBG, contourDF["contour"].values)), dtype=object)

    # Export
    contourDF.to_csv(Path(output_dir) / "{}.contour_data.csv".format(prefix), index=False)
    contourDF.to_json(Path(output_dir) / "{}.contour_data.json".format(prefix))

    print("[{}] Contour data exported to".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    print("\t{}".format(Path(output_dir) / "{}.contour_data.csv".format(prefix)))
    print("\t{}".format(Path(output_dir) / "{}.contour_data.json".format(prefix)))
   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def render_contour_plots(image, border_contour, contours, prefix, dpi=300, output_dir="./", color = (0, 255, 255), contour_thickness=3, imgFormat="pdf"):
    """Creates two contour plots: 1) border and interior contours overlaid on image 2) border and interior
    contours overlaid on image with contour indices for reference.

    Parameters
    ----------
    img : <numpy.ndarray> Query image
    border_contour : <np.ndarray> Border contour to plot
    contours : <np.ndarray> Interior contours to plot
    moments : <list> List of contour moments
    prefix : <str> Prefix for output files (e.g. prefix.noindex.pdf, prefix.pdf)
    dpi : <int> Output image resolution. Default=300
    output_dir : <str> Path to output directory. Default='./'
    """
    matplotlib.use("PDF")
    # Get figure size
    if type(image)==str:
        try:
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except cv2.error:
            image = vector2cv2(image.as_posix())

    height, width, _ = image.shape
    figsize = width / float(dpi), height / float(dpi)

    # No index image
    canvas = image.copy()
    _, ax = plt.subplots(ncols=1, figsize=figsize)
    if border_contour:
        cv2.drawContours(canvas, contours=border_contour, contourIdx=-1, color=(255,0,0), thickness=contour_thickness)
    cv2.drawContours(canvas, contours=contours, contourIdx=-1, color=color, thickness=contour_thickness)
    ax.imshow(canvas)
    ax.axis('off')
    plt.savefig(Path(output_dir) / "{}.noindex.{}".format(prefix,imgFormat), dpi=dpi, transparent=True)
    plt.close()

    # Indexed image
    canvas = image.copy()
    _, ax = plt.subplots(ncols=1, figsize=figsize)
    if border_contour:
        cv2.drawContours(canvas, contours=border_contour, contourIdx=-1, color=(255,0,0), thickness=contour_thickness)
    # Plot indexed contours
    cv2.drawContours(canvas, contours=contours, contourIdx=-1, color=color, thickness=contour_thickness)
    for i, c in enumerate(contours):
        cX,cY = contour_xy(c)
        ax.text(x=cX, y=cY, s=u"{}".format(i), color="black", size=5)

    ax.imshow(canvas)
    ax.axis('off')
    plt.savefig(Path(output_dir) / "{}.{}".format(prefix, imgFormat), dpi=dpi, transparent=True)
    plt.close()
    print("[{}] Contour plots saved to".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    print("\t{}".format(Path(output_dir) / "{}.noindex.{}".format(prefix, imgFormat)))
    print("\t{}".format(Path(output_dir) / "{}.index.{}".format(prefix, imgFormat)))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_image(image_path, neighborhood=10, prefix=None, Amin=50, Amax=10e6, detectScaleBar=False,
                  output_dir="./", print_plots=True, dpi=300, debug=False, **kwargs):
    """Parameters
    ----------
    image <str> or <numpy.ndarray> : Query image. If string, is assumed to be an image filepath; if numpy.ndarray,
        assumed to be in cv2 or numpy format.
    Amin <int> : Minimum contour area in pixels
    Amax <int> : Maximum contour area in pixels
    sliding_window <bool> : Use sliding window contour detection. Default=True
    neighborhood <int> : Neighborhood size in pixels determining a unique contour
    prefix <str> : New prefix for output files. By default the new files will reflect the input file's basename
    output_dir <str> : Path to output directory. Default='./'
    debug <bool> : writes debugging information and plots each step
    **kwargs : kwargs for `mcf`
    """

    if debug:
        print("[{}] Working directory: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), os.getcwd()))
        print("[{}] Output directory: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), output_dir))
    input_path = Path(image_path)
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

    if detectScaleBar:
        scaleBarInfo = get_scalebar_info(denoise)
        if len(scaleBarInfo) == 2:
            scaleBar, scaleBarPixelLength = scaleBarInfo
            scaleBarUnits = None
            unitsPerPixel= None
            print("[{}] Detected scale bar but could not read units. Try detection or manual drawing in MCF-GUI.".format())
            return
        elif len(scaleBarInfo) == 4:
            scaleBar, scaleBarPixelLength, scaleBarUnitLength, scaleBarUnits = scaleBarInfo
            unitsPerPixel = scaleBarUnitLength/scaleBarPixelLength
        else:
            scaleBarUnits = None
            unitsPerPixel = None
            print("[{}] Could not find scale bar. Try detection or manual drawing in MCF-GUI.".format())
            return
        cv2.line(image, (scaleBar[0], scaleBar[1]), (scaleBar[2], scaleBar[3]), (0,255,0),5)
    else:
        scaleBarUnits = None
        unitsPerPixel = None

       
    print("[{}] Getting image border...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    border_contour = mcf(image=denoise, extract_border=True, **kwargs)

    print("[{}] Finding contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    contours = mcf(denoise, skip_flood=False, progress_bar=True, **kwargs)
    contours = contour_size_selection(contours, Amin=Amin, Amax=Amax)
    contours = remove_redundant_contours(contours, neighborhood=neighborhood)
    contour_DF = pd.DataFrame(columns=["uuid4", "contour", "C", "kBlur", "blocksize", "kLaplacian", "kDilate", "kGradient", "kForeground"], dtype=object)
    contour_DF["contour"] = contours
    contour_DF["uuid4"] = [uuid.uuid4().hex for i in range(len(contour_DF))]
    contour_DF["C"] = kwargs['C']
    contour_DF["kBlur"] = kwargs['k_blur']
    contour_DF["blocksize"] = kwargs['blocksize']
    contour_DF["kLapacian"] = kwargs['k_laplacian']
    contour_DF["kDilate"] = kwargs['k_dilate']
    contour_DF["kGradient"] = kwargs['k_gradient']
    contour_DF["kForeground"] = kwargs['k_foreground']
    print("[{}] Found {} contours".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), len(contours)))
    print("[{}] Exporting contour data...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    # Export contours
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    if unitsPerPixel and scaleBarUnits:
        export_contour_data(image=image, contourDF=contour_DF, conversion_factor=unitsPerPixel,
                        units=scaleBarUnits, prefix=prefix, output_dir=output_dir, get_color=False)
    else:
        export_contour_data(image=image, contourDF=contour_DF, prefix=prefix, output_dir=output_dir, get_color=False)
    if print_plots:
        print("[{}] Plotting...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
        render_contour_plots(image=image, border_contour=border_contour, contours=contour_DF['contour'].values, 
        prefix=prefix, dpi=dpi, output_dir=output_dir)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def detect_scalebar(image, rho=1, theta=float(np.pi/180), min_votes=15, min_line_length=100,
                 max_line_gap=0, threshold1=50, threshold2=150, largest=True):
    """Detects line-like (i.e. a straight bar instead of a ruler) scalebars.
    Written by StackOverflow user SHEN and amended by Ian Gilman
    https://stackoverflow.com/a/45560545/6317380

    Parameters
    ----------
    rho : <float> Distance resolution in pixels of the Hough grid. Default = 1
    theta : <float> Angular resolution in radians of the Hough grid. Default = np.pi / 180
    min_votes : <int> Minimum number of votes (intersections in Hough grid cell). Default = 15
    min_line_length : <int> Minimum number of pixels making up a line. Default = 100
    max_line_gap : <int> Maximum gap in pixels between connectable line segments. Default = 0
    threshold1 : <int> First threshold for the hysteresis procedure. Default = 50
    threshold2 : <int> Second threshold for the hysteresis procedure. Default = 150
    color : <tuple> Color as BGR tuple for plotting. Default = (255,0,0)
    largest : <bool> Only return largest scalebar

    Returns
    -------
    line_image : <numpy.ndarray> Resulting image of detected scalebar lines
    """

    line_image = image.copy()*0  # creating a blank to draw lines on
    edges = cv2.Canny(image.copy(), threshold1=threshold1, threshold2=threshold2)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(image=edges.copy(), rho=rho, theta=theta, threshold=min_votes,
                            lines=np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    scalebars = np.array([0,0,0,0])
    scalebar_length = 0.0
    # Get largest scalebar
    if largest:
        for l in lines:
            x1 = l[0][0]
            x2 = l[0][2]
            y1 = l[0][1]
            y2 = l[0][3]

            length = np.sqrt((x2-x1)**2+(y2-y1)**2)

            if length>scalebar_length:
                scalebar_length = length
                scalebars = l[0]
        # Add scalebar to image
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    else:
        scalebars = lines.copy()
        length = None
        for line in np.array(scalebars):
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    return line_image, scalebars, length

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_units(scalebar_img):
    """Reads the units and metric of scale bar from an image and uses the number
    of pixels in the scale bar to convert pixel area to units detected in the image.

    Parameters
    ----------
    scalebar : <list>, Scale bar endpoints [x1, y1, x2, y2]
    scarebar_img : <numpy.ndarray>, Image of scale bar

    Returns
    -------
    pixel_area : <float>, The area of a single pixel in the new units
    converstion_units : <str>, The detected units"""

    scalebar_length, scalebar_units = image_to_string(image=scalebar_img.copy(), lang="eng").split()

    # Check for units in micrometers
    eng_units = image_to_string(image=scalebar_img.copy(), lang='eng').split()[1]
    grc_units = image_to_string(image=scalebar_img.copy(), lang='grc').split()[1]
    if (grc_units[0]==u"\u03BC") and (eng_units in ["um", "pm"]):
        scalebar_units = grc_units[0]+eng_units[1:]

    scalebar_length = float(scalebar_length)

    return scalebar_length, scalebar_units

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def vector2cv2(file_path):
    """Converts a vector image to a BGR cv2 format."""
    return np.array(Image.open(file_path))[:,:,::-1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_scalebar_info(image, plot=False, **kwargs):
    """Detects and reads a scalebar.

    Parameters
    ----------
    image : <np.ndarray> Image
    plot : <bool> Plot resulting scalebar. Default = False
    **kwargs : Kwargs to pass to `detect_scalebar`

    Returns
    -------
    conversion_factor : <float> Conversion factor from pixel area to new unit area (e.g. 0.1 um^2/pixel)
    units : <str> New units"""

    clone = image.copy()
    # Detect largest scalebar
    try:
        line_image, scalebar, length_in_pixels = detect_scalebar(clone, **kwargs)
    except TypeError:
        print("Could not detect scalebar.")
        return
    # Add scalebar to image
    line_edges = cv2.addWeighted(src1=clone, alpha=1.0, src2=line_image.copy(), beta=10, gamma=0)
    height, width = line_edges.shape[:2]

    # Easiest to read scalebar text when isolated. Start with small window around scalebar.
    pad = 50
    while pad <= max(height, width):
        try:
            xmin = scalebar[0]
            xmax = scalebar[2]
            ymin = scalebar[1]
            ymax = scalebar[3]
            crop_scalebar = line_edges.copy()[max([0, ymin-pad]):min([height, ymax+pad]), max([0, xmin-pad]):min([width, xmax+pad])]
            length_in_units, scalebar_units = read_units(scalebar_img=crop_scalebar)
            if plot:
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(crop_scalebar)
                plt.tight_layout()

            return scalebar, length_in_pixels, length_in_units, scalebar_units

        # If no units were found, expand search window.
        except ValueError:
            pad = pad*2
    print("Detected scalebar but could not read units.")
    return scalebar, length_in_pixels
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def RectangleOverlapTest(image, contours, x, y, width, height, REMOVE=False):
    """Finds contours that overlap with a rectangle.

    Parameters
    ----------
    image <numpy.ndarray> : Image which contours are from
    contours <list>: A list of contours
    x <int> : Left rectangle side
    y <int> : Top rectangle side
    width <int> : Rectangle width
    height <int> : Rectangle height

    Returns
    -------
    selected <list> : A list of contours
    """
    blank = np.zeros(image.shape[0:2])
    rectangle = cv2.rectangle(blank.copy(), (x, y), (x + width, y + height), 1, cv2.FILLED)
    rect_x, rect_y = (x + (width / 2), y + (height / 2))
    if REMOVE:
        rm_idx = []
        for i, c in tqdm(enumerate(contours), desc='RectangleOverlapTest', leave=False):
            (min_c_x, min_c_y), min_c_r = cv2.minEnclosingCircle(c)
            if abs(min_c_x - rect_x) > (width / 2) + (min_c_r): continue
            if abs(min_c_y - rect_y) > (height / 2) + (min_c_r): continue

            current = cv2.drawContours(blank.copy(), contours, i, 1, cv2.FILLED)
            overlap = np.logical_and(current, rectangle)
            if overlap.any():
                rm_idx.append(i)
        
        return [i for j, i in enumerate(contours) if j not in rm_idx]

    else:
        selected = []
        for i, c in tqdm(enumerate(contours), desc='RectangleOverlapTest', leave=False):
            (min_c_x, min_c_y), min_c_r = cv2.minEnclosingCircle(c)
            if abs(min_c_x - rect_x) > (width / 2) + (min_c_r): continue
            if abs(min_c_y - rect_y) > (height / 2) + (min_c_r): continue

            current = cv2.drawContours(blank.copy(), contours, i, 1, cv2.FILLED)
            overlap = np.logical_and(current, rectangle)
            if overlap.any():
                selected.append(c)
        return selected

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ContourOverlapTest(image, contours, background_contours, return_overlapping=True):
    """finds contours that overlap with a set of background contours.

    Parameters
    ----------
    image <numpy.ndarray> : Image which contours are from
    contours <list> : List of contours of interest
    background_contours  <list> : List of background contours to look for overlap with
    return_overlapping <bool> : Return overlapping contours. If False, will return contours that
        do not overlap. Default=True

    Returns
    -------
    selected <list> : A list of contours that are overlapping
    """
    selected = []
    blank = np.zeros(image.shape[0:2])
    background_image = cv2.drawContours(blank.copy(), background_contours, -1, 1, cv2.FILLED)
    for i in tqdm(range(len(contours)), desc='ContourOverlapTest', leave=False):
        current = cv2.drawContours(blank.copy(), contours, i, 1, cv2.FILLED)
        if return_overlapping:
            if np.logical_and(current, background_image).any():
                selected.append(contours[i])
        else:
            if not np.logical_and(current, background_image).any():
                selected.append(contours[i])

    return selected

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cv2pixmap(image):
    """Creates a QPixmap object from a cv2 image"""
    if type(image) == str and Path(image).exists():
        return QPixmap(image)
    elif type(image) == np.ndarray:
        cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = cv2_image.shape
        totalBytes = cv2_image.nbytes
        bytesPerLine = int(totalBytes / h)
        qimage_item = QImage(cv2_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return QPixmap(qimage_item)
    else:
        print("ERROR: Could not parse image.")
        sys.exit()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class NumpyEncoder(json.JSONEncoder):
    """Written by StackOverflow user karlB
    https://stackoverflow.com/a/47626762/6317380"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def contour2json(contours: list, file_path: Union[str, PosixPath]):
    """Saves contours to json using NumpyEncoder"""
    if type(file_path) == 'pathlib.PosixPath':
        file_path = file_path.as_posix()
    
    json.dump(obj=contours, fp=codecs.open(file_path, 'w+', encoding="utf-8"), cls=NumpyEncoder)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_json_contours(file_path: Union[str, PosixPath]) -> list:
    """Reads contours from a json"""
    if type(file_path) == 'pathlib.PosixPath':
        file_path = file_path.as_posix()
    
    json_load = json.load(codecs.open(file_path, 'r', encoding="utf-8"))

    return [np.array(c, dtype='int32') for c in json_load]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def contourScatter(x, y, contours, color, linewidth=1, alpha=1.0, ax=None, zoom=1.0):
    """A modified scatter plot that allows plotting a set of points that
    define a line as a marker. Written by Stackoverflow user Joe Kington 
    and modified for contour use by HansHirse.
    https://stackoverflow.com/a/22570069/11089932

    Parameters
    ----------
    x <iterable> : x-positions
    y <iterable> : y-positions
    contours <iterable of numpy.ndarray> : Contours to plot as markers
    color <str, tuple, or iterable> : Matplotlib color string or color in RGB or RGBA 
        format, or an iterable of those types
    linewidth <int> : Contour line width. If -1, will fill shape with <color>
    alpha <float> : Marker transparency
    ax <matplotlib.axes> : Axes to use
    zoom <float> : Marker scale
    
    Returns
    -------
    artists <list> : List of plot attributes"""

    if ax is None:
        ax = plt.gca()
        
    x, y = np.atleast_1d(x, y)
    
    if type(color)==str or np.shape(color) in [(3,), (4,)]:
        color = np.reshape(np.tile(rgba2bgra(color=color), len(x)), (len(x),4))
    else:
        if len(color)==len(x):
            color = [rgba2bgra(c) for c in color]
        else:
            print("Length of color does not match number of contours.")
        color = [rgba2bgra(c) for c in color]

    artists = []
    for x0, y0, cont, col in zip(x, y, contours, color):
        cx, cy, cw, ch = cv2.boundingRect(cont)
        img = np.zeros((ch + 11, cw + 11, 4), np.uint8)
        img = cv2.drawContours(img, [cont], -1, col, linewidth, offset=(-cx+5, -cy+5))
        img = OffsetImage(img, zoom=zoom, alpha=alpha)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    
    return artists

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def rgba2bgra(color):
    if type(color)==str: # String to BGRA255
        try:
            color = np.array(matplotlib.colors.to_rgba(color))[[2,1,0,3]]*255
        except ValueError:
            print("Invalid matplotlib color")
            return 
    elif np.shape(color) == (3,): #RGB to RGBA
        color = np.array(color)[::-1]*255
        color = np.append(color,255)
    elif np.shape(color) == (4,):
        color = np.array(color)[[2,1,0,3]]*255
    else:
        print("Invalid matplotlib color")
        return
    return color