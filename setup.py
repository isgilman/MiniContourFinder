import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MiniContourFinder",
    version="1.0.17",
    author="Ian S Gilman",
    author_email="ian.gilman@yale.edu",
    description="Lightweight image segmentation software for biological images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isgilman/MiniContourFinder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Intended Audience :: Science/Research",
    ],
    install_requires=[
        "matplotlib >= 3.1.1",
        "numpy > 1.20.3",
        "opencv-python >= 4.5.2",
        "pandas >= 0.25.2",
        "pillow >= 6.2.1",
        "Pillow",
        "pyperclip >= 1.7.0",
        "PyQt5 >= 5.12.3",
        "pytesseract >= 0.3.0",
        "scipy >= 1.6.3",
        "tqdm >= 4.36.1"
    ],
    entry_points={
        'console_scripts':[
            'mcf=MCF.mcf:main',
            'mcf_gui=MCF.mcf_gui:main',
            'mcf_parallel=MCF.mcf_parallel:main'
            ]}
)