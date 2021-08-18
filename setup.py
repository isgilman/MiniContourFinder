import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MiniContourFinder",
    version="1.0.7",
    author="Ian S Gilman",
    author_email="ian.gilman@yale.edu",
    description="Lightweight image segmentation software for biological images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isgilman/MiniContourFinder",
    packages=["MCF"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "opencv-python >= 4.5.2",
        "pillow >= 6.2.1",
        "Pillow",
        "pyperclip >= 1.7.0",
        "PyQt5 >= 5.12.3",
        "pytesseract >= 0.3.0",
        "tqdm >= 4.36.1"
    ]
)