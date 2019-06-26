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
