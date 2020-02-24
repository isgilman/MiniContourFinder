# Measuring cross sections with computers
See the `README`.

## 0 Setting up shop
```bash
➜  ~ $ conda create -yn compvision python=3
➜  ~ source activate compvision
(compvision) ➜  ~ $ conda install -c conda-forge opencv
(compvision) ➜  ~ $ conda install numpy pandas scipy matplotlib jupyter_core
(compvision) ➜  ~ $ pip install imutils
(compvision) ➜  ~ $ cd /Dropbox/GitHub_repos/CrossSection_DeepLearning/Notebooks
(compvision) ➜  Notebooks git:(detection_sandbox) ✗ $ conda install -c conda-forge pillow
```

You can install python modules within a jupyter notebook using

```python
!{sys.executable} -m pip install <module>
```

or

```python
!conda install --yes --prefix {sys.prefix} <module>
```

See [here](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/) for more.

I used this to install `pytesseract` within the correct kernel.

```python
!{sys.executable} -m pip install pytesseract
```

We also need to install `tesseract`.

```bash
(compvision) ➜  ~ $ brew install tesseract
```

## 19 December 2019
### Progress
- Contour detecting mostly works for many images
- Classifying things as cell or non-cell is working mostly with contour size and overlap with convex hull

### Problems
- Many images that seem like they should work well don't work at all
  - This includes images where I've reduced file size from Gb to Mb scale, and they are pretty unwieldy in their larger form
- It's difficult to accurately classify shapes with high precision
  - Epidermal and vascular cells are tough
- Sizes of objects changes a lot with image size so window sizes, expected contour sizes, etc cannot be uniform
  - Not sure how much this is affecting other areas like contour detection itself but I know that it is to some extent. For example, the size of kernels needs to change with image size but it isn't clear how to set sizes relative to the image.
- Currently we're measuring all cells but we need to focus on photosynthetic cells when making these measurements
- Airspace is much harder to pull out of images than cells, so we need to rethink how to measure airspace

### Short term goals
- Optimize existing code so we can redo analyses and generate new data quickly for building/refining other parts
  - Get Jupyter notebook into CLI form should help with this but we still have to troubleshoot individual images
- Do some manual tuning to get a number of high quality images done to build datasets for exploration and classification
  - **Look into [scikit-image](https://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html) and maybe try [non-local means](https://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html?highlight=non%20local%20means) for blurring**
- Explore contour space
  - What happens if we try to classify contours based on a few simple metrics (e.g. area, perimeter, area:perimeter, area:hull, approximate polygon)? Maybe a classification scheme naturally falls out or we'll at least get a handle on where variation is
  - Need to export contour measurements with index on image or maybe moment will get this done
  - **Look at [contour properties](https://docs.opencv.org/master/d1/d32/tutorial_py_contour_properties.html)**

### Long term goals
- Almost any image should be able to be fed in
- Some sort of GUI would be nice because we're working with images, but I hesitate because a shitty GUI is worse than no GUI at all
  - This could even be to select the part of the image to measure
- Classification of cell types
  - Photosynthetic, hydrenchyma, vascular, epidermal
  - May rely on position
- Get someone to help with theory behind what we're doing
  - Why do we use the kernel sizes, processing steps, etc? Can we hard code parameter tuning based on some measures of the input image?
  - Possibly could feed in a set of standard images with a range of parameter values to understand behavior better, but in my experience the dynamics are not so predictable


# GUI building
## Py GTK+ 3
For `pygtk` I need to install `jhbuild`

```bash
(compvision) ➜  apps $ cd ~/apps
(compvision) ➜  apps $ git clone https://gitlab.gnome.org/GNOME/jhbuild.git
(compvision) ➜  apps $ cd jhbuild
(compvision) ➜  jhbuild git:(master) $ ./autogen.sh
(compvision) ➜  jhbuild git:(master) $ make
(compvision) ➜  jhbuild git:(master) $ make install
(compvision) ➜  jhbuild git:(master) $ PATH=$PATH:~/.local/bin
```

The last line temporarily adds the `jhbuild` tools to my path. Note that I did not have `yelp-tools`. So the install used a plain Makefile:
> If gnome-common, yelp-tools and autotools are not available, autogen.sh will configure JHBuild to install via a plain Makefile.

```bash
(compvision) ➜  ~ $ cd ~/Dropbox/GitHub_repos/CrossSection_DeepLearning
(compvision) ➜  CrossSection_DeepLearning $ jhbuild sanitycheck
libtool >= 1.5 not found
aclocal cant see libtool macros
Please copy the lacking macros (libtool.m4) in one of the following paths: /Users/iangilman/jhbuild/install/share/aclocal, /usr/share/aclocal, /Users/iangilman/miniconda3/envs/compvision/share/aclocal
Could not find DocBook XML DTD V4.1.2 in XML catalog (usually part of package 'docbook-xsl')
Could not find DocBook XSL Stylesheets in XML catalog (usually part of package 'docbook-xsl')
Could not find the Perl module XML::Parser (usually part of package 'libxml-parser-perl' or 'perl-XML-Parser')
dbus-python not found
(compvision) ➜  CrossSection_DeepLearning $ conda install libtool perl-XML-Parser dbus-python
(compvision) ➜  CrossSection_DeepLearning $ brew install docbook docbook-xsld
```

Still receiving the errors about DocBook, so installing with `sudo` and MacPorts:

```bash
(compvision) ➜  ~ $ sudo port install libxslt docbook-xsl docbook-xml-4.2
```

## PyQT

```bash
(compvision) ➜  ~ $ conda install -c conda-forge pyqt
```
