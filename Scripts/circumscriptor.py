#!/usr/bin/env python
# coding: utf-8

# core
import sys, os, re, argparse
import numpy as np
from datetime import datetime
try: import pandas as pd
except ImportError:
    print("\n\tError: pandas is not installed/loaded")
    sys.exit()
try: from scipy import spatial
except ImportError:
    print("\n\tError: scipy is not installed/loaded")
    sys.exit()
try: from tqdm import tqdm
except ImportError:
    print("\n\tError: tqdm is not installed/loaded")
    sys.exit()
# plotting
import matplotlib.pyplot as plt
# image recognition
try: import cv2
except ImportError:
    print("\n\tError: cv2 is not installed/loaded")
    sys.exit()
try: import imutils
except ImportError:
    print("\n\tError: imutils is not installed/loaded")
    sys.exit()
try:
    import pytesseract
    from pytesseract import image_to_string
except ImportError:
    print("\n\tError: pytesseract is not installed/loaded")
    sys.exit()
try:
    import skimage.measure
    from skimage import io, data
    from skimage.util import img_as_float, img_as_ubyte
except ImportError:
    print("\n\tError: skimage is not installed/loaded")
    sys.exit()
try:
    from pathlib import Path
except ImportError:
    print("\n\tError: pathlib is not installed/loaded")
    sys.exit()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)() # retain local pointer to value
        return value                     # faster to return than dict lookup

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def flatten(l):
    """Flattens a list of sublists"""
    return [item for sublist in l for item in sublist]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def flood_fill(image):
    """Flood fill from the edges."""
    mirror = image.copy()
    height, width = image.shape[:2]
    for row in range(height):
        if mirror[row, 0] == 255:
            cv2.floodFill(mirror, None, (0, row), 0)
        if mirror[row, width-1] == 255:
            cv2.floodFill(mirror, None, (width-1, row), 0)

    for col in tqdm(range(width), desc='Flooding background', leave=False):
        if mirror[0, col] == 255:
            cv2.floodFill(mirror, None, (col, 0), 0)
        if image[height-1, col] == 255:
            cv2.floodFill(mirror, None, (col, height-1), 0)

    return mirror

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mcf(image, k_blur=9, C=3, blocksize=15, k_laplacian=5, k_dilate=5, k_gradient=3, k_foreground=7,
        extract_border=False, skip_flood=False, debug=False):
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
    if debug: print("Gray...")
    gray = cv2.cvtColor(src = image, code=cv2.COLOR_RGB2GRAY)
    """Adaptive histogram normalization"""
    gridsize = int(min(0.01*max(image.shape[:2]), 8))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    hequalized = clahe.apply(gray)
    """Blur"""
    if debug: print("Blur...")
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
        flood = flood_fill(foreground)
    if extract_border:
        """Get border"""
        if debug: print("Getting border...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        background = cv2.dilate(flood, kernel, iterations=2)
        background_image, background_contours, background_hierarchy = cv2.findContours(image=background.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        border_contour = [max(background_contours, key=cv2.contourArea)]
        return border_contour

    """Find contours"""
    if debug: print("Drawing contours...")
    new_img, contours, hierarchy = cv2.findContours(image=flood.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    return contours

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def contour_size_selection(contours, pmin=50, pmax=10e5, Amin=50, Amax=10e6):
    """Selections contours based on perimeter and area

    Parameters
    ----------
    contours : <list> A list of contours
    pmin : <int> 50; Minimum perimeter in pixels
    pmax : <int> 10e5; Maximum perimeter in pixels
    Amin : <int> 50; Minimum area in pixels
    Amax : <int> 10e6; Maximum area in pixels

    Returns
    -------
    large_contours : <list> A list of contours
    """

    large_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, closed=True)
        if (pmax >= perimeter >= pmin) and (Amax >= area >= Amin):
            large_contours.append(c)

    return large_contours

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
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sliding_contour_finder(image, stepsize, winW, winH, neighborhood, border_contour, skip_flood=False, debug=False, **kwargs):
    """Uses a sliding-window approach to find contours across a large image. Uses KDTree algorithm to
    remove duplicated contours from overlapping windows.

    Parameters
    ----------
    image : <numpy.ndarray> Query image
    stepsize : <int> Slide step size in pixels (currently the same in x and y directions)
    winW : <int> Window width in pixels
    winH : <int> Window height in pixels
    neighborhood : <int> Neighborhood size in pixels determining a unique contour
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
    moments = []
    for i, (x,y,window) in tqdm(enumerate(windows), total=n_windows, desc='Windows'):
        if debug: print(("Window {}, x0: {}, y0: {}, shape: {}".format(i,x,y,np.shape(window))))
        if window.shape[0] != winH or window.shape[1] != winW: continue
        if window.sum() == 0: continue
        """Running mini contour finder in window"""
        if debug: print("Running mini contour finder...")
        window_contours = mcf(window, skip_flood=skip_flood,  **kwargs)
        if debug: print("Found {} contours in window {}".format(len(window_contours), i))

        """Remove overlapping contours"""
        if debug: print("Refining contours...")
        for c in window_contours:
            c[:,:,0] += x
            c[:,:,1] += y

            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int((M["m10"] / M["m00"])) # moment X
                cY = int((M["m01"] / M["m00"])) # moment Y
            else:
                cX,cY = 0,0

            if len(moments)==0:
                contours.append(c)
                moments.append([cX, cY])
            else: # if previous moments exist, find the distance and index of the nearest neighbor
                distance, index = spatial.KDTree(moments).query([cX, cY])
                if distance > neighborhood: # add point if moment falls outside of neighborhood
                    contours.append(c)
                    moments.append([cX, cY])

    if debug: print("Found {} non-overlapping contours".format(len(contours)))

    return contours

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def refine_contours(image, contours, border_contours, overlap_thresh=0, pmin=50, pmax=10e5, Amin=50, Amax=10e6,
                    min_cell_area=100, check_intersection=True, separate_cells=False):

    """Size select contours"""
    init_large_contours = contour_size_selection(contours, pmin, pmax, Amin, Amax)

    """Get smoothed contours and contour hulls"""
    init_smooth_contours, init_hulls = smooth_contours(init_large_contours)

    """Create image of positive space"""
    blank = np.zeros(image.shape[0:2])
    border_image = cv2.drawContours(blank.copy(), border_contours, 0, 1, -1)

    cell_contours = []
    air_contours = []
    fin_smooth_contours = []
    for i, (c, h) in tqdm(enumerate(zip(init_smooth_contours, init_hulls)), total=len(init_smooth_contours)):
        """Identify cells with overlap criterion"""
        if separate_cells:
            c_area, h_area = cv2.contourArea(c), cv2.contourArea(h)
            if (c_area >= overlap_thresh*h_area) and (c_area >= min_cell_area):
                cell_contours.append(c)
            else:
                air_contours.append(c)
        else:
            fin_smooth_contours.append(c)
    if separate_cells:
        return cell_contours, air_contours
    else:
        return fin_smooth_contours

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def detect_scalebar(image, rho=1, theta=float(np.pi/180), min_votes=15, min_line_length=100,
                 max_line_gap=0, threshold1=50, threshold2=150, color=(255,0,0), largest=True):
    """Detects line-like (i.e. a straight bar instead of a ruler) scalebars.

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
def read_units(scalebar, scalebar_img):
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

    scalebar_pixels = np.sqrt((scalebar[2]-scalebar[0])**2+(scalebar[3]-scalebar[1])**2)
    scalebar_length, scalebar_units = image_to_string(image=scalebar_img.copy(), lang="eng").split()

    # Check for units in micrometers
    eng_units = image_to_string(image=scalebar_img.copy(), lang='eng').split()[1]
    grc_units = image_to_string(image=scalebar_img.copy(), lang='grc').split()[1]
    if (grc_units[0]==u"\u03BC") and (eng_units in ["um", "pm"]):
        scalebar_units = grc_units[0]+eng_units[1:]

    scalebar_length = float(scalebar_length)

    pixel_area = (scalebar_length**2)/(scalebar_pixels**2)
    converstion_units = "{}^2".format(scalebar_units)

    return pixel_area, converstion_units


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_scalebar_info(image, plot=False, debug=False, **kwargs):
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
        line_image, scalebar, length = detect_scalebar(clone, **kwargs)
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
            conversion_factor, units = read_units(scalebar=scalebar, scalebar_img=crop_scalebar)
            if plot:
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(crop_scalebar)
                plt.tight_layout()

            return conversion_factor, units

        # If no units were found, expand search window.
        except ValueError:
            pad = pad*2
    print("Detected scalebar but could not read units.")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def measure_image(image, c_cells, c_airspace, c_border, conversion_factor=None, units=None, use_pixels=False):
    """Calculates a variety of metrics about cell size and intercellular air space in pixels
    and with units if available.

    Parameters
    ----------
    image : <np.ndarray> Image
    c_cells : <list> A list of cell contours
    c_airspace : <list> A list of airspace contours
    c_border : <list> A list of border contours
    conversion_factor : <float> Conversion factor from pixel area to new unit area (e.g. 0.1 um^2/pixel)
    units : <str> New units; Default = None
    use_pixels : <bool> Return without units; Default = False

    Returns
    -------
    A_cells(_pixels) : <list> A list of the cell areas corresponding to c_cells
    A_airspace_(pixels) : <list> A list of the intercellular areas corresponding to c_airspace
    """

    if not use_pixels:
        clone = image.copy()
        if not (conversion_factor) and (units):
            conversion_factor, units = get_scalebar_info(image=clone)

    A_cells_pixels = np.array([float(cv2.contourArea(c)) for c in c_cells])
    A_airspace_pixels = np.array([float(cv2.contourArea(c)) for c in c_airspace])
    A_border_pixels = float(cv2.contourArea(c_border[0]))
    IAS_bg_pixels = A_border_pixels - sum(A_cells_pixels)
    IAS_as_pixels = sum(A_airspace_pixels)
    IAS_fraction_bg_pixels = IAS_bg_pixels/A_border_pixels
    IAS_fraction_as_pixels = IAS_as_pixels/A_border_pixels

    if use_pixels:
        print("Using border area")
        print("\tAverage cell area: {:.2f}".format(np.mean(A_cells_pixels)))
        print("\tTotal cell area: {:.2f}".format(np.sum(A_cells_pixels)))
        print("\tborder area: {:.2f}".format(A_border_pixels))
        print("\tIAS: {:.3f} ({:.3f}%)".format(IAS_bg_pixels, IAS_fraction_bg_pixels))
        print("Using airspace contours")
        print("\tIAS: {:.3f} ({:.3f}%)".format(IAS_as_pixels, IAS_fraction_as_pixels))
        return A_cells_pixels, A_airspace_pixels


    A_cells = A_cells_pixels*conversion_factor
    A_airspace = A_airspace_pixels*conversion_factor
    A_border = A_border_pixels*conversion_factor
    IAS_bg = IAS_bg_pixels*conversion_factor
    IAS_as = IAS_as_pixels*conversion_factor
    IAS_fraction_bg = IAS_bg/A_border
    IAS_fraction_as = IAS_as/A_border

    print("Using border area")
    print("\tAverage cell area: {:.2f} {}".format(np.mean(A_cells), units))
    print("\tTotal cell area: {:.2f} {}".format(np.sum(A_cells), units))
    print("\tborder area: {:.2f} {}".format(A_border, units))
    print("\tIAS: {:.3f} {} ({:.3f}%)".format(IAS_bg, units, IAS_fraction_bg))
    print("Using airspace contours")
    print("\tIAS: {:.3f} {} ({:.3f}%)".format(IAS_as, units, IAS_fraction_as))
    return A_cells, A_airspace

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
        unselected = np.delete(contours, rm_idx)
        return list(unselected)
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

def export_contour_data(image, contours, prefix, conversion_factor=None, units=None, output_dir="./"):
    """Exports relevant data on contours in CSV and pickle format.

    Parameters
    ----------
    image : <np.ndarray> Image
    contours : <np.ndarray> Contours to export
    conversion_factor : <float> Conversion factort to convert pixel area. Default=None
    units : <str> Associated conversion factor units. Default=None
    prefix : <str> Prefix for output files (e.g. PREFIX.contour_data.pkl, PREFIX.contour_data.csv)
    output_dir : <str> Path to output directory. Default='./' """

    if type(image)==str:
        image = cv2.imread(image)

    DF = pd.DataFrame()
    DF["contour"] = contours
    DF["moments"] = [cv2.moments(c) for c in contours]
    DF["area"] = [cv2.contourArea(c) for c in contours]

    X = []
    Y = []
    cXY = []
    cR = []
    bboxH = []
    bboxW = []
    for c in contours:
        # Get position
        M = cv2.moments(c)
        if M["m00"] != 0:
            x = int((M["m10"] / M["m00"]))-10
            y = int((M["m01"] / M["m00"]))
        else:
            x,y = 0,0
        X.append(x)
        Y.append(y)
        # Minimum circle
        min_c_xy, min_c_r = cv2.minEnclosingCircle(c)
        cXY.append(min_c_xy)
        cR.append(min_c_r)
        # Bounding box
        bbx, bby, bbw, bbh = cv2.boundingRect(c)
        bboxH.append(bbh)
        bboxW.append(bbw)

    DF["xy"] = list(zip(X,Y))
    DF["min_circle_xy"] = cXY
    DF["min_circle_r"] = cR
    DF["bbox_height"] = bboxH
    DF["bbox_width"] = bboxW
    DF["aspect_ratio"] = DF["bbox_width"]/DF["bbox_height"]
    DF["bbox_area"] = DF["bbox_width"]*DF["bbox_height"]
    DF["extent"] = DF["area"]/DF["bbox_area"]
    DF["convex_hull"] = [cv2.convexHull(c) for c in contours]
    DF["convexity"] = [cv2.isContourConvex(c) for c in contours]
    DF["solidity"] = [float(cv2.contourArea(c))/cv2.contourArea(cv2.convexHull(c)) for c in contours]
    DF["equivalent_d"] = [np.sqrt(4*cv2.contourArea(c)/np.pi) for c in contours]

    """Get mean color"""
    R = []
    G = []
    B = []
    for i, c in enumerate(contours):
        clone = image.copy()
        mask = np.zeros_like(clone)  # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, contours, i, 255, -1)  # Draw filled contour in mask
        out = np.zeros_like(image)  # Extract out the object and place into output image
        out[mask == 255] = image[mask == 255]
        pixelpoints = np.transpose(np.nonzero(mask))

        r, g, b = image[flatten(pixelpoints[:, :2])[::2], flatten(pixelpoints[:, :2])[1::2]].mean(axis=0) / 255
        R.append(r)
        G.append(g)
        B.append(b)

    DF["mean_R"] = R
    DF["mean_G"] = G
    DF["mean_B"] = B


    if conversion_factor and units:
        DF["area_{}".format(units)] = DF["area_pixels"]*conversion_factor

    # Export summary statistics
    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    numeric = DF[DF.columns[is_number(DF.dtypes)]]
    summary = numeric.describe()
    summary.to_csv(Path(output_dir) / "{}.summary.csv".format(prefix))
    # Export contours and individual metrics
    DF.to_csv(Path(output_dir) / "{}.contour_data.csv".format(prefix))
    DF.to_pickle(Path(output_dir) / "{}.contour_data.pkl".format(prefix))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def render_contour_plots(image, border_contour, contours, prefix, dpi=300, output_dir="./", color = (0, 255, 255), contour_thickness=3):
    """Creates two contour plots: 1) border and interior contours overlaid on image 2) border and interior
    contours overlaid on image with contour indices for reference.

    Parameters
    ----------
    img : <numpy.ndarray> Query image
    border_contour : <np.ndarray> Border contour to plot
    contours : <np.ndarray> Interior contours to plot
    moments : <list> List of contour moments
    prefix : <str> Prefix for output files (e.g. prefix.noindex.tif, prefix.tif)
    dpi : <int> Output image resolution. Default=300
    output_dir : <str> Path to output directory. Default='./'
    """
    # Get figure size
    if type(image)==str:
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, depth = image.shape
    figsize = width / float(dpi), height / float(dpi)

    # No index image
    canvas = image.copy()
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    cv2.drawContours(canvas, contours=border_contour, contourIdx=-1, color=color, thickness=contour_thickness)
    cv2.drawContours(canvas, contours=contours, contourIdx=-1, color=color, thickness=contour_thickness)
    ax.imshow(canvas)
    ax.axis('off')
    # plt.tight_layout()
    plt.savefig(Path(output_dir) / "{}.noindex.tif".format(prefix), dpi=dpi, transparent=True)
    plt.close()

    # Indexed image
    canvas = image.copy()
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    if border_contour:
        cv2.drawContours(canvas, contours=border_contour, contourIdx=-1, color=color, thickness=contour_thickness)
    # Plot indexed contours
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int((M["m10"] / M["m00"]))-10
            cY = int((M["m01"] / M["m00"]))
        else:
            cX,cY = 0,0
        cv2.drawContours(canvas, contours=[c], contourIdx=-1, color=color, thickness=contour_thickness)
        ax.text(x=cX, y=cY, s=u"{}".format(i), color="black", size=8)

    ax.imshow(canvas)
    ax.axis('off')
    plt.savefig(Path(output_dir) / "{}.tif".format(prefix), dpi=dpi, transparent=True)
    plt.close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_image(image_path, neighborhood=10, prefix=None, stepsize=500, winW=1000, winH=1000, Amin=50, Amax=10e6, sliding_window=True,
                  output_dir="./", border_contour='DETECT', print_plots=True, dpi=300, debug=False, **kwargs):
    """Parameters
    ----------
    image <str> or <numpy.ndarray> : Query image. If string, is assumed to be an image filepath; if numpy.ndarray,
        assumed to be in cv2 or numpy format.
    stepsize <int> : Slide step size in pixels (currently the same in x and y directions)
    winW <int> : Window width in pixels
    winH <int> : Window height in pixels
    Amin <int> : Minimum contour area in pixels
    Amax <int> : Maximum contour area in pixels
    sliding_window <bool> : Use sliding window contour detection. Default=True
    neighborhood <int> : Neighborhood size in pixels determining a unique contour
    prefix <str> : New prefix for output files. By default the new files will reflect the input file's basename
    output_dir <str> : Path to output directory. Default='./'
    border_contour <np.ndarray> : Border contour. If set to 'DETECT', the method `mcf` will
        be used to find the border contour. Default=`DETECT`
    debug <bool> : writes debugging information and plots each step
    **kwargs : kwargs for `mcf`
    """

    if debug:
        print("[{}] Working directory: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), os.getcwd()))
        print("[{}] Output directory: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), output_dir))
    input_path = Path(image_path)
    image = cv2.imread(str(input_path))
    if not prefix:
        prefix = input_path.stem
    if debug:
        print("[{}] Input file: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), input_path.absolute()))

    """Denoise"""
    print("[{}] Denoising image...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    image = cv2.fastNlMeansDenoisingColored(image.copy(), None, 10, 10, 7, 21)
    if border_contour != 'DETECT':
        if debug: print("[{}] Border contour given; skipping border detection".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    else:
        print("[{}] Getting border...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
        border_contour = mcf(image=image, extract_border=True, **kwargs)

    print("[{}] Finding contours...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    if sliding_window:
        contours = sliding_contour_finder(image=image.copy(), stepsize=stepsize, winW=winW,
                 winH=winH, border_contour=border_contour, neighborhood=neighborhood, **kwargs)
    else:
        contours = mcf(image=image, **kwargs)
    contours = contour_size_selection(contours, Amin=Amin, Amax=Amax)
    print("[{}] Found {} contours".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), len(contours)))
    print("[{}] Exporting contour data...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    # Export contours
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    export_contour_data(image=image, contours=contours, conversion_factor=None,
                        units=None, prefix="{}.ALL".format(prefix), output_dir=output_dir)

    print("[{}] Contour data exported to".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
    print("\t{}".format(Path(output_dir) / "{}.summary.csv".format(prefix)))
    print("\t{}".format(Path(output_dir) / "{}.contour_data.csv".format(prefix)))
    print("\t{}".format(Path(output_dir) / "{}.contour_data.pkl".format(prefix)))

    if print_plots:
        print("[{}] Plotting...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
        render_contour_plots(image=image, border_contour=border_contour, contours=contours, prefix=prefix, dpi=300, output_dir=output_dir)

        print("[{}] Contour plots saved to".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
        print("\t{}".format(Path(output_dir) / "{}.tif".format(prefix)))
        print("\t{}".format(Path(output_dir) / "{}.noindex.tif".format(prefix)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def query_yes_no(question, default):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    print("""
               ╭━━╮         ╭╮
╭━━┳┳┳━┳━━┳┳┳━━┫ ━╋━━┳┳━┳┳━━┫╰┳━━┳┳━╮
┃ ━┫┃╭━┫ ━┫┃┃┃┃┣━ ┃ ━┫╭━┫┃  ┃╭┫  ┃╭━╯
╰━━┻┻╯ ╰━━┻━┻┻┻┻━━┻━━┻╯ ╰┫╭━┻━┻━━┻╯
                         ╰╯
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

    args = parser.parse_args()

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


    print("[{}] Circumscriptor command:\n\t python circumscriptor.py --prefix {} --sliding-window {} --stepsize {} --winW {} --winH {} --neighborhood {} --dpi {} --debug {} --k_blur {} --C {} --blocksize {} --k_laplacian {} --k_dilate {} --k_gradient {} --k_foreground {} --Amin {} --Amax {:d}".format(
        datetime.now().strftime('%d %b %Y %H:%M:%S'), args.prefix, args.sliding_window, stepsize, winW, winH, args.neighborhood, args.dpi, args.debug, args.k_blur, args.C, args.blocksize, args.k_laplacian, args.k_dilate, args.k_gradient, args.k_foreground, int(args.Amin), int(args.Amax)))

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

    process_image(image_path=input_path, prefix=prefix, sliding_window=args.sliding_window, stepsize=stepsize, winW=winW, winH=winH,
                  Amin=args.Amin, Amax=args.Amax, neighborhood=args.neighborhood, output_dir=output_dir_path,
                  dpi=args.dpi, debug=args.debug, **kwargs)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print("[{}] Time elapsed: {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), end-start))