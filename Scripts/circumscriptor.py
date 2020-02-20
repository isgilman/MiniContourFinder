#!/usr/bin/env python
# coding: utf-8

# core
import sys, os, re, argparse, csv
from argparse import ArgumentParser
import numpy as np
from glob import glob
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
    gray = cv2.cvtColor(src = image.copy(), code=cv2.COLOR_RGB2GRAY)
    """Blur"""
    if debug: print("Blur...")
    blur = cv2.GaussianBlur(src=gray, ksize=(k_blur, k_blur), sigmaX=2)
    """Adaptive Gaussian threshold"""
    if debug: print("Adapt Gauss Thresh...")
    thresh = cv2.adaptiveThreshold(src=blur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=blocksize, C=C)

    """The next commented out blocks relate to morphological snakes"""
#     if extract_border:
#         """Get border"""
#         if debug: print("Getting border...")
#         # Morphological GAC
#         sk_compat = img_as_float(thresh)
#         gimage = inverse_gaussian_gradient(sk_compat)
#         # Initial level set
#         init_ls = np.zeros(shape=sk_compat.shape, dtype=np.int8)
#         init_ls[10:-10, 10:-10] = 1
#         # List with intermediate results for plotting the evolution
#         print("GAC")
#         ls = morphological_geodesic_active_contour(gimage=gimage, iterations=200, init_level_set=init_ls,
#                                         smoothing=1, balloon=-1, threshold=0.9)#, iter_callback=callback)
#         # Convert back to cv2
#         background = img_as_ubyte(ls)
#         print("Get contours")
#         background_image, background_contours, background_hierarchy = cv2.findContours(image=background.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#         border_contour = [max(background_contours, key=cv2.contourArea)]
#         return border_contour

    """Laplacian"""
    if debug: print("Laplacian...")
    laplacian = cv2.Laplacian(src=thresh, ddepth=cv2.CV_16S, ksize=k_laplacian)
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
    tozero = cv2.threshold(dilate, 127, 255, cv2.THRESH_TOZERO)
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
        unknown = cv2.subtract(background, foreground)
        background_image, background_contours, background_hierarchy = cv2.findContours(image=background.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        border_contour = [max(background_contours, key=cv2.contourArea)]
        return border_contour
    """The next commented out blocks relate to morphological snakes"""
#         # Morphological GAC
#         sk_compat = img_as_float(thresh)
#         gimage = inverse_gaussian_gradient(sk_compat)

#         # Initial level set
#         init_ls = np.zeros(shape=sk_compat.shape, dtype=np.int8)
#         init_ls[10:-10, 10:-10] = 1
#         # List with intermediate results for plotting the evolution
# #         evolution = []
# #         callback = store_evolution_in(evolution)
#         ls = morphological_geodesic_active_contour(gimage=gimage, iterations=200, init_level_set=init_ls,
#                                         smoothing=1, balloon=-1, threshold=0.9)#, iter_callback=callback)
#         # Convert back to cv2
#         background = img_as_ubyte(ls)
#         background_image, background_contours, background_hierarchy = cv2.findContours(image=background.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#         border_contour = [max(background_contours, key=cv2.contourArea)]
#         return border_contour

    """Find contours"""
    if debug: print("Drawing contours...")
    new_img, contours, hierarchy = cv2.findContours(image=flood.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        drawn_contours = cv2.drawContours(image=image.copy(), contours=contours, contourIdx=-1, color=(0, 0, 255),
                                          thickness=2)
        """Get background"""
        if debug: print("Getting background...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        background = cv2.dilate(flood, kernel, iterations=2)
        unknown = cv2.subtract(background, foreground)

        """Watershed"""
        print("Watershed...")
        markers = cv2.connectedComponents(foreground)[1]
        markers += 1  # Add one to all labels so that background is 1, not 0
        markers[unknown==255] = 0  # mark the region of unknown with zero
        markers = cv2.watershed(image, markers)
        """Create marker image"""
        print("Creating marker image...")
        hue_markers = np.uint8(179*np.float32(markers)/np.max(markers))
        blank_channel = 255*np.ones((h, w), dtype=np.uint8)
        marker_img = cv2.merge([hue_markers, blank_channel, blank_channel])
        marker_img = cv2.cvtColor(marker_img, cv2.COLOR_HSV2BGR)

        """Plotting"""
        print("Plotting...")
        fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(24, 24), sharex=True, sharey=True)

        ax[0,0].imshow(gray, cmap="gray")
        ax[0,0].set_title("Gray")
        ax[0,1].imshow(blur, cmap="gray")
        ax[0,1].set_title("Gaussian blur")
        ax[0,2].imshow(thresh, cmap="gray")
        ax[0,2].set_title("Gaussian Adaptive Threshold")
        ax[1,0].imshow(laplacian, cmap="gray")
        ax[1,0].set_title("Laplacian")
        ax[1,1].imshow(dilate, cmap="gray")
        ax[1,1].set_title("Dilate")
        ax[1,2].imshow(gradient, cmap="gray")
        ax[1,2].set_title("Gradient")
        ax[2,0].imshow(binary, cmap="gray")
        ax[2,0].set_title("Binarized")
        ax[2,1].imshow(foreground, cmap="gray")
        ax[2,1].set_title("Clean up foreground")
        ax[2,2].imshow(flood, cmap="gray")
        ax[2,2].set_title("Flood fill")
        ax[3,0].imshow(background, cmap="gray")
        ax[3,0].set_title("Background")
        ax[3,1].imshow(marker_img, cmap="gray")
        ax[3,1].set_title("Marker image")
        ax[3,2].imshow(drawn_contours, cmap="gray")
        ax[3,2].set_title("Drawn contours")
        for i in range(4):
            for j in range(3):
                ax[i,j].axis("off")
        plt.tight_layout(pad=0)
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
    large_smooth = []
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
    smoothed, hulls = smooth_contours(contours=contours)

    return contours, smoothed, hulls

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

def export_contour_data(contours, moments, prefix, conversion_factor=None, units=None, output_dir="./"):
    """Exports relevant data on contours in CSV and pickle format.

    Parameters
    ----------
    contours : <np.ndarray> Contours to export
    moments : <list> List of contour moments
    conversion_factor : <float> Conversion factort to convert pixel area. Default=None
    units : <str> Associated conversion factor units. Default=None
    prefix : <str> Prefix for output files (e.g. PREFIX.contour_data.pkl, PREFIX.contour_data.csv)
    output_dir : <str> Path to output directory. Default='./' """

    DF = pd.DataFrame()
    DF["contour"] = contours
    DF["moments"] = moments
    DF["area_pixels"] = [cv2.contourArea(c) for c in contours]

    if conversion_factor and units:
        DF["area_{}".format(units)] = DF["area_pixels"]*conversion_factor

    DF.to_csv(os.path.join(output_dir, "{}.contour_data.csv".format(prefix)))
    DF.to_pickle(os.path.join(output_dir, "{}.contour_data.pkl".format(prefix)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def render_contour_plots(img, border_contour, contours, prefix, dpi=300, output_dir="./", figsize=(16,16)):
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
    figsize : <tuple> Output figure size. Default=(16,16)
    """

    # No index image
    canvas = img.copy()
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    cv2.drawContours(canvas, contours=border_contour, contourIdx=-1, color=(255,0,0), thickness=2)
    cv2.drawContours(canvas, contours=contours, contourIdx=-1, color=(0,255,255), thickness=2)
    ax.imshow(canvas)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "{}.noindex.tif".format(prefix)), dpi=dpi, transparent=True)
    plt.close()

    # Indexed image
    canvas = img.copy()
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    cv2.drawContours(canvas, contours=border_contour, contourIdx=-1, color=(255,0,0), thickness=2)
    # Plot indexed contours
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int((M["m10"] / M["m00"]))-10
            cY = int((M["m01"] / M["m00"]))
        else:
            cX,cY = 0,0
        cv2.drawContours(canvas, [c], -1, (0, 255, 255), 2)
        ax.text(x=cX, y=cY, s=u"{}".format(i), color="black", size=8)

    ax.imshow(canvas)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "{}.tif".format(prefix)), dpi=dpi, transparent=True)
    plt.close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_image(filepath, neighborhood=10, prefix=None, stepsize=500, winW=1000, winH=1000, Amin=50, Amax=10e6,
                  image=None, output_dir="./", border_contour='DETECT', print_plots=True, dpi=300, figsize=(16,16),
                  debug=False, **kwargs):
    """Parameters
    ----------
    filepath : <str> Filepath to query image.
    stepsize : <int> Slide step size in pixels (currently the same in x and y directions)
    winW : <int> Window width in pixels
    winH : <int> Window height in pixels
    Amin : <int> Minimum contour area in pixels
    Amax : <int> Maximum contour area in pixels
    neighborhood : <int> Neighborhood size in pixels determining a unique contour
    prefix : <str> New prefix for output files. By default the new files will reflect the input file's basename
    image : <numpy.ndarray> Query image. Note that is only for explicitly reading an
        image in cv2 or skimage format, NOT an image from a filepath. Default=None
    output_dir : <str> Path to output directory. Default='./'
    border_contour : <np.ndarray> Border contour. If set to 'DETECT', the method `mcf` will
        be used to find the border contour. Default=`DETECT`
    figsize : <tuple> Output figure size. Default=(16,16)
    debug : <bool> writes debugging information and plots each step
    **kwargs : kwargs for `mcf`
    """

    if debug:
        print("[{}] Working directory: {}".format(datetime.now(), os.getcwd()))
        print("[{}] Output directory: {}".format(datetime.now(), output_dir))
    if filepath:
        img = cv2.imread(filepath)
        if not prefix:
            prefix = ".".join(re.split("\.", os.path.basename(filepath))[:-1])
        if debug:
            print("[{}] Input file: {}".format(datetime.now(), filepath))
    if image:
        if debug: print("Working with image not from file...")
        img = image.copy()

    if border_contour != 'DETECT':
        if debug: print("[{}] Border contour given; skipping border detection".format(datetime.now()))
    else:
        print("[{}] Getting border...".format(datetime.now()))
        border_contour = mcf(image=img, extract_border=True, **kwargs)

    print("[{}] Finding contours...".format(datetime.now()))
    contours, s_contours, contour_hulls = sliding_contour_finder(image=img.copy(), stepsize=stepsize, winW=winW,
               winH=winH, border_contour=border_contour, neighborhood=neighborhood, **kwargs)
    contours = contour_size_selection(contours, Amin=Amin, Amax=Amax)
    print("Found {} contours".format(len(contours)))
    print("[{}] Exporting contour data...".format(datetime.now()))
    # Export all contours
    export_contour_data(contours=contours, moments=[cv2.moments(c) for c in contours], conversion_factor=None,
                        units=None, prefix="{}.ALL".format(prefix), output_dir=output_dir)
    print("[{}] Contour data exported to".format(datetime.now()))
    print(os.path.join(output_dir, "{}.contour_data.csv".format("{}.ALL".format(prefix))))
    print(os.path.join(output_dir, "{}.contour_data.pkl".format("{}.ALL".format(prefix))))

    if print_plots:
        print("[{}] Plotting...".format(datetime.now()))
        render_contour_plots(img=img, border_contour=border_contour, contours=contours, prefix=prefix, dpi=300,
                             output_dir=output_dir, figsize=figsize)

        print("[{}] Plot saved to\n\t{} and\n\t{}".format(datetime.now(),
                                                          os.path.join(output_dir, "{}.tif".format(prefix)),
                                                          os.path.join(output_dir, "{}.noindex.tif".format(prefix))))

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
    parser.add_argument("input", type=str,
                    help="Filepath to query image")
    parser.add_argument("output_dir", type=str, default='./',
                    help="Path to output directory. Default='./'")
    parser.add_argument("--prefix", type=str,
                    action="store", dest="prefix",
                    help="New prefix for output files. By default the new files will reflect the input file's basename")
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
    parser.add_argument("--figsize", type=tuple, default=(16,16),
                    action="store", dest="figsize",
                    help="Output figure size. Default=(16,16)")
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
        prefix = ".".join(re.split("\.", os.path.basename(args.input))[:-1])
    else:
        prefix = args.prefix


    print("[{}] Circumscriptor command:\n\t python circumscriptor.py --prefix {} --stepsize {} --winW {} --winH {} --neighborhood {} --figsize {} --dpi {} --debug {} --k_blur {} --C {} --blocksize {} --k_laplacian {} --k_dilate {} --k_gradient {} --k_foreground {} --Amin {} --Amax {}".format(
        datetime.now(), args.prefix, stepsize, winW, winH, args.neighborhood, args.figsize, args.dpi, args.debug, args.k_blur, args.C, args.blocksize, args.k_laplacian, args.k_dilate, args.k_gradient, args.k_foreground, args.Amin, args.Amax))
    print("[{}] Input file: {}".format(datetime.now(), args.input))
    print("[{}] Output directory: {}".format(datetime.now(), args.output_dir))

    if os.path.exists(os.path.join(args.output_dir, prefix+".ALL.contour_data.csv")):
        print("\tFound pickle for {}. Delete output files to rerun analysis".format(prefix))
        sys.exit()

    process_image(filepath=args.input, prefix=prefix, stepsize=stepsize, winW=winW, winH=winH,
                  Amin=args.Amin, Amax=args.Amax, neighborhood=args.neighborhood, output_dir=args.output_dir,
                  dpi=args.dpi, debug=args.debug, **kwargs)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print("[{}] Time elapsed: {}".format(datetime.now(), end-start))