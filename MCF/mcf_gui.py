#!/usr/bin/env python
# coding: utf-8

# Core
import sys, cv2, uuid
from datetime import datetime
import pyperclip
import numpy as np
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
# PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmapCache
# Plotting
import matplotlib
if sys.platform == 'darwin':
    matplotlib.use("TkAgg")
# Custom utilities
try:
    from helpers import *
    from imagetools import *
except ModuleNotFoundError:
    from MCF.helpers import *
    from MCF.imagetools import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize window
        self.setWindowTitle('MCF - {}'.format(sys.argv[1]))
        self.resize(1000, 400)
        # Add toolbar with menu
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        saveFile = QAction("&Save", self)
        saveFile.setShortcut("Ctrl+S")
        saveFile.setStatusTip("Save File")
        saveFile.triggered.connect(self.saveAll)
        toolbar.addAction(saveFile)

        openFile = QAction("&Open", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(self.file_open)
        toolbar.addAction(openFile)

        # Initialize widget with layout
        widget = QWidget()
        layout = QVBoxLayout(widget)
        # Set contour app as central widget
        self.contour_app = ContourApp(self)
        layout.addWidget(self.contour_app)
        self.setCentralWidget(widget)
        self.contour_app.update_plot()
        self.contour_app.viewer.updateView()

        self.show()
     
    def saveAll(self):
        saveDialog = QFileDialog.getSaveFileName(parent=self, 
        caption="File prefix",
        directory=Path(Path(sys.argv[1]).parent / Path(sys.argv[1]).stem).as_posix())

        if saveDialog[0] != '':
            if len(self.contour_app.contour_DF) == 0 :
                export_data = pd.DataFrame(np.zeros(shape=(len(self.contour_app.large_contours), 9)), 
                columns=["uuid4", "contour", "C", "kBlur", "blocksize", "kLaplacian","kDilate", "kGradient", "kForeground"], dtype=object)
                export_data["contour"] = self.contour_app.large_contours
                export_data["uuid4"] = [uuid.uuid4().hex for i in range(len(self.contour_app.large_contours))]
                export_data["C"] = self.contour_app.C.value()
                export_data["kBlur"] = self.contour_app.k_blur.value()
                export_data["blocksize"] = self.contour_app.blocksize.value()
                export_data["kLaplacian"] = self.contour_app.k_laplacian.value()
                export_data["kDilate"] = self.contour_app.k_dilate.value()
                export_data["kGradient"] = self.contour_app.k_gradient.value()
                export_data["kForeground"] = self.contour_app.k_foreground.value()
            else:
                export_data = self.contour_app.contour_DF

            export_contour_data(image=sys.argv[1], contourDF=export_data, conversion_factor=self.contour_app.unitsPerPixel, units=self.contour_app.scaleBarUnits,
                                output_dir=Path(saveDialog[0]).parent, prefix=Path(saveDialog[0]).stem)
            render_contour_plots(image=sys.argv[1], contours=export_data["contour"].values, border_contour=None, contour_thickness=self.contour_app.contour_thickness.value(),
                                output_dir=Path(saveDialog[0]).parent, prefix=Path(saveDialog[0]).stem, color=self.contour_app.highlight_color)
        else: return

    def file_open(self):
        fileFilter = 'Contour data file (*.json)'
        openDialog = QFileDialog.getOpenFileName(self, 
            caption = "Open contour data file (.json)",
            directory = Path(Path(sys.argv[1]).parent / Path(sys.argv[1]).stem).as_posix(),
            filter = fileFilter)

        if openDialog[0] != '':
            self.contour_app.select_contours.setChecked(True)
            loaded = pd.read_json(openDialog[0])
            loaded = loaded[self.contour_app.contour_DF.columns]
            loaded['contour'] = loaded['contour'].apply(np.array)
            print("[{}] Loaded {} contours from {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), len(loaded), openDialog[0]))
            self.contour_app.contour_DF = self.contour_app.contour_DF.append(loaded)
            self.contour_app.update_plot()
        else:return

    def closeEvent(self, event):
        """Generate 'question' dialog on clicking 'X' button in title bar.

        Reimplement the closeEvent() event handler to include a 'Question'
        dialog with options on how to proceed - Save, Close, Cancel buttons
        """
        reply = QMessageBox.question(
            self, "Message",
            "Are you sure you want to quit? Any unsaved work will be lost.",
            QMessageBox.Save | QMessageBox.Close | QMessageBox.Cancel,
            QMessageBox.Save)

        if reply == QMessageBox.Close:
            event.accept()
        elif reply == QMessageBox.Save:
            self.file_save()
        else:
            event.ignore()

class ContourApp(QWidget):
    """See the following StackOverflow post for embedding a QGraphicsView in
    another widget:
    https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview"""
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        ### Create window with grid layout ###
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.click_x, self.click_y = (0,0)
        self.setMouseTracking(True)

        ### Initialize viewer ###
        self.viewer = GraphicsView(self)
        self.grid_layout.addWidget(self.viewer, 0, 0, -1, 8) # Note that grid coords are (row, col, rowspan, colspan)
        self.viewer.updateView()
        self.viewer.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.viewer.setAlignment(QtCore.Qt.AlignTop)

        ### Add image and contours ###
        self.image_path = Path(sys.argv[1])
        image = cv2.imread(self.image_path.as_posix())
        if image is None:
            image = vector2cv2(self.image_path.as_posix())

        wdir = self.image_path.parent
        results = wdir.glob("{}.denoise.*".format(self.image_path.stem))
        try:
            self.denoised_path = next(results)
            self.cv2_image = cv2.imread(self.denoised_path.as_posix())
            print("[{}] Found existing denoised file ({})".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), self.denoised_path.as_posix()))
        except StopIteration:
            print("[{}] Denoising image...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
            self.cv2_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            self.denoised_path = Path("{}/{}.denoise{}".format(wdir.as_posix(), self.image_path.stem, ".png"))
            cv2.imwrite(filename=self.denoised_path.as_posix(), img=self.cv2_image)

            print("[{}] Created temporary denoised file ({})".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), self.denoised_path.as_posix()))

        # Get contours
        self.contours = mcf(image=self.cv2_image)
        self.contour_color = (255, 0, 0)
        self.highlight_color = (255, 0, 255)
        self.contour_DF = pd.DataFrame(columns=["uuid4", "contour", "C", "kBlur", "blocksize", "kLaplacian", "kDilate", "kGradient", "kForeground"], dtype=object)
        cv2.drawContours(self.cv2_image, contours=self.contours, contourIdx=-1, color=self.contour_color, thickness=3)
        if len(self.contour_DF)>0:
            cv2.drawContours(self.cv2_image, contours=self.contour_DF["contour"].values, contourIdx=-1, color=self.highlight_color, thickness=3)

        # Convert to Qimage object
        self.pixmap = cv2pixmap(self.cv2_image)
        self.viewer.setPhoto(self.pixmap)
        self.viewer.updateView()

        ### Create sliders ###
        self.k_blur = QSlider(QtCore.Qt.Horizontal)
        self.k_blur.setRange(2, 26)
        self.k_blur.setValue(5)
        self.k_blur.setTickInterval(1)
        self.k_blur.setTickPosition(QSlider.TicksBelow)
        self.blur_label = QLabel()
        self.blur_label.setText('k_blur: {}'.format(2*self.k_blur.value()-1))
        self.grid_layout.addWidget(self.blur_label, 0, 10)
        self.grid_layout.addWidget(self.k_blur, 0, 8, 1, 2)

        self.C = QSlider(QtCore.Qt.Horizontal)
        self.C.setRange(1, 25)
        self.C.setValue(3)
        self.C.setTickInterval(1)
        self.C.setTickPosition(QSlider.TicksBelow)
        self.C_label = QLabel()
        self.C_label.setText('C: {}'.format(self.C.value()))
        self.grid_layout.addWidget(self.C_label, 1, 10)
        self.grid_layout.addWidget(self.C, 1, 8, 1, 2)

        self.blocksize = QSlider(QtCore.Qt.Horizontal)
        self.blocksize.setRange(2, 200)
        self.blocksize.setValue(8)
        self.blocksize.setTickInterval(1)
        self.blocksize.setTickPosition(QSlider.TicksBelow)
        self.blocksize_label = QLabel()
        self.blocksize_label.setText('blocksize: {}'.format(2*self.blocksize.value()-1))
        self.grid_layout.addWidget(self.blocksize_label, 2, 10)
        self.grid_layout.addWidget(self.blocksize, 2, 8, 1, 2)

        self.k_laplacian = QSlider(QtCore.Qt.Horizontal)
        self.k_laplacian.setRange(2, 15)
        self.k_laplacian.setValue(3)
        self.k_laplacian.setTickInterval(1)
        self.k_laplacian.setTickPosition(QSlider.TicksBelow)
        self.laplacian_label = QLabel()
        self.laplacian_label.setText('k_laplacian: {}'.format(2*self.k_laplacian.value()-1))
        self.grid_layout.addWidget(self.laplacian_label, 3, 10)
        self.grid_layout.addWidget(self.k_laplacian, 3, 8, 1, 2)

        self.k_dilate = QSlider(QtCore.Qt.Horizontal)
        self.k_dilate.setRange(2, 52)
        self.k_dilate.setValue(3)
        self.k_dilate.setTickInterval(1)
        self.k_dilate.setTickPosition(QSlider.TicksBelow)
        self.dilate_label = QLabel()
        self.dilate_label.setText('k_dilate: {}'.format(2*self.k_dilate.value()-1))
        self.grid_layout.addWidget(self.dilate_label, 4, 10)
        self.grid_layout.addWidget(self.k_dilate, 4, 8, 1, 2)

        self.k_gradient = QSlider(QtCore.Qt.Horizontal)
        self.k_gradient.setRange(2, 52)
        self.k_gradient.setValue(2)
        self.k_gradient.setTickInterval(1)
        self.k_gradient.setTickPosition(QSlider.TicksBelow)
        self.gradient_label = QLabel()
        self.gradient_label.setText('k_gradient: {}'.format(2*self.k_gradient.value()-1))
        self.grid_layout.addWidget(self.gradient_label, 5, 10)
        self.grid_layout.addWidget(self.k_gradient, 5, 8, 1, 2)

        self.k_foreground = QSlider(QtCore.Qt.Horizontal)
        self.k_foreground.setRange(2, 15)
        self.k_foreground.setValue(4)
        self.k_foreground.setTickInterval(1)
        self.k_foreground.setTickPosition(QSlider.TicksBelow)
        self.foreground_label = QLabel()
        self.foreground_label.setText('k_foreground: {}'.format(2*self.k_foreground.value()-1))
        self.grid_layout.addWidget(self.foreground_label, 6, 10)
        self.grid_layout.addWidget(self.k_foreground, 6, 8, 1, 2)

        self.sliders = [self.k_blur, self.C, self.blocksize, self.k_laplacian, self.k_dilate, self.k_gradient, self.k_foreground]

        ### Set contour size ###
        self.Amin = QLineEdit()
        self.Amin.setPlaceholderText("Minimum contour area")
        self.setAmin = QPushButton("Enter")
        self.grid_layout.addWidget(self.Amin, 7, 8, 1, 2)
        self.grid_layout.addWidget(self.setAmin, 7, 10)

        self.Amax = QLineEdit()
        self.Amax.setPlaceholderText("Maximum contour area")
        self.setAmax = QPushButton("Enter")
        self.grid_layout.addWidget(self.Amax, 8, 8, 1, 2)
        self.grid_layout.addWidget(self.setAmax, 8, 10)

        ### Scale bar info
        self.scaleBar = None
        self.scaleBarUnits = None
        self.scaleBarPixelLength = None
        self.scaleBarUnitLength = None
        self.unitsPerPixel = None

        self.pixelLengthField = QLineEdit()
        self.pixelLengthField.setPlaceholderText("Length in pixels")
        self.grid_layout.addWidget(self.pixelLengthField, 9, 8, 1, 2)
        self.pixelLabel = QLabel("Pixels")
        self.grid_layout.addWidget(self.pixelLabel, 9, 10, 1, 1)

        self.unitLengthField = QLineEdit()
        self.unitLengthField.setPlaceholderText("Length in units")
        self.grid_layout.addWidget(self.unitLengthField, 10, 8, 1, 2)
        self.unitsLabel = QLabel("Length")
        self.grid_layout.addWidget(self.unitsLabel, 10, 10, 1, 1)

        self.setScaleButton = QPushButton("Set scale")
        self.grid_layout.addWidget(self.setScaleButton, 11, 8, 1, 1)

        self.detectScaleButton = QPushButton("Detect scale")
        self.grid_layout.addWidget(self.detectScaleButton, 11, 9, 1, 1)

        self.clearScaleButton = QPushButton("Clear scale")
        self.grid_layout.addWidget(self.clearScaleButton, 12, 8, 1, 1)

        self.drawing = False
        self.drawScaleButton = QPushButton("Draw scale bar")
        self.grid_layout.addWidget(self.drawScaleButton, 12, 9, 1, 1)

        ### Generate parameters for CLI ###
        self.CLIgenerator = QPushButton()
        self.CLIgenerator.setText("Copy CLI parameters")
        self.grid_layout.addWidget(self.CLIgenerator, 11, 10, 1, 1)

        ### Text box ###
        self.box = QPlainTextEdit()
        self.box.insertPlainText("--k_blur 9 --C 3 --blocksize 15 --k_laplacian 5 --k_dilate 5 --k_gradient 3 --k_foreground 7 --Amin 50 --Amax 10e6")

        ### Reset parameters ###
        self.resetParams = QPushButton()
        self.resetParams.setText("Reset parameters")
        self.grid_layout.addWidget(self.resetParams, 12, 10, 1, 1)

        ### Color picker ###
        self.contour_colorPicker = QPushButton()
        self.contour_colorPicker.setText("Contour color")
        self.grid_layout.addWidget(self.contour_colorPicker, 13, 8, 1, 1)

        self.highlight_colorPicker = QPushButton()
        self.highlight_colorPicker.setText("Selected color")
        self.grid_layout.addWidget(self.highlight_colorPicker, 13, 9, 1, 1)

        ### Contour thickness ###
        self.contour_thickness = QSlider(QtCore.Qt.Horizontal)
        self.contour_thickness.setRange(1, 10)
        self.contour_thickness.setValue(3)
        self.contour_thickness.setTickInterval(1)
        self.contour_thickness.setTickPosition(QSlider.TicksBelow)
        self.thickness_label = QLabel()
        self.thickness_label.setText('Contour thickness: {}'.format(self.contour_thickness.value()))
        self.grid_layout.addWidget(self.thickness_label, 14, 10)
        self.grid_layout.addWidget(self.contour_thickness, 14, 8, 1, 2)

        ### Contour selection ###
        self.select_contours = QCheckBox("Select contours")
        self.grid_layout.addWidget(self.select_contours, 13, 10, 1, 1)
        self.select_contours.stateChanged.connect(self.selectContoursChecked)

        ### Approximate polygons ###
        self.use_approxPolys = QCheckBox("Approximate polygons (epsilon)")
        self.grid_layout.addWidget(self.use_approxPolys, 15, 8, 1, 1)
        self.epsilon = QSlider(QtCore.Qt.Horizontal)
        self.epsilon.setRange(1, 20)
        self.epsilon.setValue(1)
        self.epsilon.setTickInterval(1)
        self.epsilon.setTickPosition(QSlider.TicksBelow)
        self.epsilon_label = QLabel()
        self.epsilon_label.setText('episilon: {}'.format(self.epsilon.value()))
        self.grid_layout.addWidget(self.epsilon_label, 16, 10)
        self.grid_layout.addWidget(self.epsilon, 16, 8, 1, 2)

        ### Convex hulls ###
        self.use_convexHulls = QCheckBox("Use convex hulls")
        self.grid_layout.addWidget(self.use_convexHulls, 15, 9, 1, 1)

        ### Connections ###
        # Sliders -> plots
        self.k_blur.valueChanged.connect(self.update_plot)
        self.C.valueChanged.connect(self.update_plot)
        self.blocksize.valueChanged.connect(self.update_plot)
        self.k_laplacian.valueChanged.connect(self.update_plot)
        self.k_dilate.valueChanged.connect(self.update_plot)
        self.k_gradient.valueChanged.connect(self.update_plot)
        self.k_foreground.valueChanged.connect(self.update_plot)
        self.contour_thickness.valueChanged.connect(self.update_plot)
        self.epsilon.valueChanged.connect(self.update_plot)
        # Sliders -> text
        self.k_blur.valueChanged.connect(self.update_text)
        self.C.valueChanged.connect(self.update_text)
        self.blocksize.valueChanged.connect(self.update_text)
        self.k_laplacian.valueChanged.connect(self.update_text)
        self.k_dilate.valueChanged.connect(self.update_text)
        self.k_gradient.valueChanged.connect(self.update_text)
        self.k_foreground.valueChanged.connect(self.update_text)
        self.contour_thickness.valueChanged.connect(self.update_text)
        self.epsilon.valueChanged.connect(self.update_text)
        
        # Line edits -> plots
        self.Amin.returnPressed.connect(self.update_plot)
        self.setAmin.clicked.connect(self.update_plot)
        self.Amax.returnPressed.connect(self.update_plot)
        self.setAmax.clicked.connect(self.update_plot)

        # Scale bars
        self.pixelLengthField.returnPressed.connect(self.setScale)
        self.unitLengthField.returnPressed.connect(self.setScale)
        self.detectScaleButton.clicked.connect(self.detectScale)
        self.clearScaleButton.clicked.connect(self.clearScale)
        self.drawScaleButton.clicked.connect(self.drawScaleBar)
        
        ### Misc. buttons ###
        # Generator push button
        self.CLIgenerator.clicked.connect(self.on_CLIgenerator_clicked)
        self.CLIgenerator.clicked.connect(self.copy_CLI)
        # Reset push button
        self.resetParams.clicked.connect(self.on_reset_clicked)
        self.resetParams.clicked.connect(self.update_text)
        self.resetParams.clicked.connect(self.update_plot)
        # Color picker button
        self.contour_colorPicker.clicked.connect(self.update_contour_color)
        self.highlight_colorPicker.clicked.connect(self.update_highlight_color)
        # Connect photo click and rubberband
        self.viewer.photoClicked.connect(self.photoClicked)
        self.viewer.rectChanged.connect(self.rectChange)
        self.update_plot()

    def drawScaleBar(self):
        if self.select_contours.isChecked():
            print("Cannot draw scale bar while selecting contours")
            return
        else:
            self.clearScale()
            self.drawing = True
            self.scaleBar = []

    def setScale(self):
        try:
            self.scaleBarPixelLength = eval(self.pixelLengthField)
            self.scaleBarUnitLength = eval(self.unitLengthField)
            self.unitsPerPixel = self.scaleBarUnitLength/self.scaleBarPixelLength
            print("Conversion factor: {} units/pixel".format(self.unitsPerPixel))
        except TypeError:
            print("Both number of pixels and a length must be set to create a scaling factor")
            return
        except NameError:
            print("Both the number of pixels and length must be numbers")
            return

    def detectScale(self):
        scaleBarInfo = get_scalebar_info(self.cv2_image)
        if len(scaleBarInfo) == 2:
            self.scaleBar, self.scaleBarPixelLength = scaleBarInfo
        elif len(scaleBarInfo) == 4:
            self.scaleBar, self.scaleBarPixelLength, self.scaleBarUnitLength, self.scaleBarUnits = scaleBarInfo
            self.unitsLabel.setText("{}".format(self.scaleBarUnits))

        self.pixelLengthField.setText("{}".format(self.scaleBarPixelLength))
        self.unitLengthField.setText("{}".format(self.scaleBarUnitLength))
        self.update_plot()

    def clearScale(self):
        self.scaleBar = None
        self.scaleBarPixelLength = None
        self.scaleBarUnitLength = None
        self.scaleBarUnits = None
        self.pixelLengthField.setText("Length in pixels")
        self.unitLengthField.setText("Length in units")
        self.unitsLabel.setText("Units")
        self.update_plot()

    def photoClicked(self, pos):
        if self.select_contours.isChecked():
            self.click_x, self.click_y = (pos.x(), pos.y())

            if self.viewer.button == 1: # Click addition
                for c in self.large_contours:
                    if cv2.pointPolygonTest(contour=c, pt=(self.click_x, self.click_y), measureDist=False) == 1.0:
                        self.contour_DF = self.contour_DF.append({
                            "uuid4": uuid.uuid4().hex, 
                            "contour": c, 
                            "C": self.C.value(), 
                            "kBlur": self.k_blur.value(),
                            "blocksize": self.blocksize.value(), 
                            "kLaplacian": self.k_laplacian.value(),
                            "kDilate": self.k_dilate.value(), 
                            "kGradient": self.k_gradient.value(), 
                            "kForeground": self.k_foreground.value()}, ignore_index=True)
                        break

            elif self.viewer.button == 2: # Click subtraction
                if len(self.contour_DF) > 0:
                    for i, row in self.contour_DF.iterrows():
                        if cv2.pointPolygonTest(contour=row["contour"], pt=(self.click_x, self.click_y), measureDist=False) == 1.0:
                            self.contour_DF = self.contour_DF.drop(index=i, axis=0)
                            self.contour_DF = self.contour_DF.reset_index(drop=True)
                            break            
            else: return
            self.update_plot()

        elif self.drawing:
            if len(self.scaleBar) == 0:
                self.scaleBar.append(pos.x())
                self.scaleBar.append(pos.y())
                self.scaleBarPixelLength = None
                self.update_plot()
            elif len(self.scaleBar) == 2:
                self.scaleBar.append(pos.x())
                self.scaleBar.append(pos.y())
                self.scaleBarPixelLength = np.sqrt((self.scaleBar[2]-self.scaleBar[0])**2+(self.scaleBar[3]-self.scaleBar[1])**2)
                self.pixelLengthField.setText("{}".format(self.scaleBarPixelLength))
                self.update_plot()

                _, okPressed = QInputDialog.getDouble(self, "Length in units", "Length:", 1, 0)
                if okPressed:
                    self.scaleBarUnitLength = _
                    _, okPressed = QInputDialog.getText(self, "Scale bar units", "Units:", QLineEdit.Normal, "\u03BC"+"m")
                    self.unitLengthField.setText(str(self.scaleBarUnitLength))
                    if okPressed and _!='':
                        self.scaleBarUnits = _
                        self.unitsLabel.setText(self.scaleBarUnits)
                else:        
                    self.unitLengthField.setText("Length in units")
                    self.unitsLabel.setText("Units")
                self.drawing = False
            else:
                self.clearScale()
                self.drawing = False        
        else:
            return

    def rectChange(self, pos):
        if self.viewer._photo.isUnderMouse() and self.select_contours.isChecked():
            xi, yi, xf, yf = (self.click_x, self.click_y, pos.x(), pos.y())
            w,h = (abs(xf-xi), abs(yf-yi))
            try:
                if (w*h) <= self.Amin:
                    return
            except TypeError:
                if (w*h) <= 4:
                    return

            if self.viewer.button == 1: # Rubberband addition
                selected = RectangleOverlapTest(image=self.cv2_image, contours=self.large_contours,
                                                x=min(xi, xf), y=min(yi, yf), width=w, height=h)
                for c in selected:
                    self.contour_DF = self.contour_DF.append({
                            "uuid4": uuid.uuid4().hex, 
                            "contour": c, 
                            "C": self.C.value(), 
                            "kBlur": self.k_blur.value(),
                            "blocksize": self.blocksize.value(), 
                            "kLaplacian": self.k_laplacian.value(),
                            "kDilate": self.k_dilate.value(), 
                            "kGradient": self.k_gradient.value(), 
                            "kForeground": self.k_foreground.value()}, ignore_index=True)

            elif self.viewer.button == 2: # Rubberband subtraction
                to_remove = RectangleOverlapTest(image=self.cv2_image, contours=self.contour_DF["contour"].values,
                                                        x=min(xi, xf), y=min(yi, yf), width=w, height=h)
                self.contour_DF = self.contour_DF[~self.contour_DF["contour"].isin(to_remove)]
                self.contour_DF = self.contour_DF.reset_index(drop=True)
            else: return
            
            self.update_plot()

    def selectContoursChecked(self):
        if self.select_contours.isChecked():
            self.viewer.selectContoursChecked = True
            for slider in self.sliders:
                slider.setEnabled(False)
        else:
            self.viewer.selectContoursChecked = False
            for slider in self.sliders:
                slider.setEnabled(True)

    def resizeEvent(self, event):
        self.update_plot()

    def update_plot(self):
        # Clear image
        QPixmapCache.clear()
        # If actively tuning contours...
        self.cv2_image = cv2.imread(self.denoised_path.as_posix())
        if not self.select_contours.isChecked():
            self.contours = mcf(image=self.cv2_image,
                                k_blur=2 * self.k_blur.value() - 1,
                                C=self.C.value(),
                                blocksize=2 * self.blocksize.value() - 1,
                                k_laplacian=2 * self.k_laplacian.value() - 1,
                                k_dilate=2 * self.k_dilate.value() - 1,
                                k_gradient=2 * self.k_gradient.value() - 1,
                                k_foreground=2 * self.k_foreground.value() - 1, skip_flood=True)
            # Set min/max contour size
            try: Amin = int(self.Amin.text())
            except ValueError: Amin = int(1)
            try: Amax = int(self.Amax.text())
            except ValueError: Amax = int(self.cv2_image.shape[0]* self.cv2_image.shape[1])
            # Refine contours
            self.large_contours = contour_size_selection(self.contours, Amin=Amin, Amax=Amax)
            if self.use_approxPolys.isChecked():
                self.large_contours = [cv2.approxPolyDP(curve=c, epsilon=self.epsilon.value(), closed=True) for c in self.large_contours]
            if self.use_convexHulls.isChecked():
                self.large_contours = [cv2.convexHull(c) for c in self.large_contours]

        cv2.drawContours(self.cv2_image, contours=self.large_contours, contourIdx=-1, color=self.contour_color, thickness=self.contour_thickness.value())
        # Add scale bar
        if self.scaleBar is not None:
            if self.drawing:
                cv2.circle(self.cv2_image, (self.scaleBar[0], self.scaleBar[1]), radius=5, color=(0,255,0), thickness=-1)
                if len(self.scaleBar) == 4:
                    cv2.circle(self.cv2_image, (self.scaleBar[2], self.scaleBar[3]), radius=5, color=(0,255,0), thickness=-1)
                    cv2.line(self.cv2_image, (self.scaleBar[0], self.scaleBar[1]), (self.scaleBar[2], self.scaleBar[3]), (0,255,0),5)
            else:
                cv2.line(self.cv2_image, (self.scaleBar[0], self.scaleBar[1]), (self.scaleBar[2], self.scaleBar[3]), (0,255,0),5)

        if len(self.contour_DF) > 0:
            cv2.drawContours(self.cv2_image, contours=self.contour_DF["contour"].values, contourIdx=-1, color=self.highlight_color, thickness=self.contour_thickness.value())

        # Convert to QImage
        self.pixmap = cv2pixmap(self.cv2_image)
        self.viewer.setPhoto(self.pixmap)
        self.viewer.updateView()

    def update_contour_color(self):
        self.contour_color = QColorDialog.getColor().getRgb()
        self.update_plot()

    def update_highlight_color(self):
        self.highlight_color = QColorDialog.getColor().getRgb()
        self.update_plot()

    def update_text(self):
        self.blur_label.setText('k_blur: {}'.format(2*self.k_blur.value()-1))
        self.C_label.setText('C: {}'.format(self.C.value()))
        self.blocksize_label.setText('blocksize: {}'.format(2*self.blocksize.value()-1))
        self.laplacian_label.setText('k_laplacian: {}'.format(2*self.k_laplacian.value()-1))
        self.dilate_label.setText('k_dilate: {}'.format(2*self.k_dilate.value()-1))
        self.gradient_label.setText('k_gradient: {}'.format(2*self.k_gradient.value()-1))
        self.foreground_label.setText('k_foreground: {}'.format(2*self.k_foreground.value()-1))
        self.thickness_label.setText('Contour thickness: {}'.format(self.contour_thickness.value()))
        self.epsilon_label.setText('epsilon: {}'.format(self.epsilon.value()))

    def on_CLIgenerator_clicked(self):
        try: Amin = int(self.Amin.text())
        except ValueError: Amin = int(1)
        try: Amax = int(self.Amax.text())
        except ValueError: Amax = int(10e6)
        self.box.clear()
        self.box.insertPlainText("--k_blur {} --C {} --blocksize {} --k_laplacian {} --k_dilate {} --k_gradient {} --k_foreground {} --Amin {} --Amax {}".format(
                2 * self.k_blur.value() - 1,
                self.C.value(),
                2 * self.blocksize.value() - 1,
                2 * self.k_laplacian.value() - 1,
                2 * self.k_dilate.value() - 1,
                2 * self.k_gradient.value() - 1,
                2 * self.k_foreground.value() - 1,
                Amin, Amax))

    def copy_CLI(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self.box.toPlainText(), mode=cb.Clipboard)
        pyperclip.copy(self.box.toPlainText())

    def on_reset_clicked(self):
        self.k_blur.setValue(5)
        self.C.setValue(3)
        self.blocksize.setValue(8)
        self.k_laplacian.setValue(3)
        self.k_dilate.setValue(3)
        self.k_gradient.setValue(2)
        self.k_foreground.setValue(4)
        self.epsilon.setValue(3)
        self.Amin.clear()
        self.Amax.clear()

class GraphicsView(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)
    rectChanged = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super().__init__(parent)    
        self._zoom = 0
        self.factor = 1.0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.button = None

        # Set up rubberband selection
        self.rubberband = QRubberBand(QRubberBand.Rectangle, self)
        self.setMouseTracking(True)
        self.origin = QtCore.QPoint()
        self.changeRubberBand = False

    def updateView(self):
        if self._zoom <= 1.0:
            scene = self.scene()
            r = scene.sceneRect()
            self.fitInView(r, QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        self.updateView()

    def setPhoto(self, pixmap=None):
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())

    def keyPressEvent(self, event):
        zoom_key = False
        if event.key() == QtCore.Qt.Key_Equal and event.modifiers() == QtCore.Qt.ControlModifier:
            self.factor = 1.1
            self._zoom += 1
            zoom_key = True
        elif event.key() == QtCore.Qt.Key_Minus and event.modifiers() == QtCore.Qt.ControlModifier:
            self.factor = 0.9
            self._zoom -= 1
            zoom_key = True
        elif event.key() == QtCore.Qt.Key_Space and event.modifiers() == QtCore.Qt.ControlModifier:
            self.factor = 1.0
            self._zoom = 0
            self.updateView()
            zoom_key = True
        if zoom_key:
            if self._zoom > 0:
                self.scale(self.factor, self.factor)
            elif self._zoom == 0:
                self.updateView()

    def mousePressEvent(self, event):
        self.button = event.button()
        if self._photo.isUnderMouse():
            self.origin = event.pos()
            self.rubberband.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
            self.changeRubberBand = True
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(GraphicsView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.changeRubberBand:
            self.rubberband.show()
            self.rubberband.setGeometry(QtCore.QRect(self.origin, event.pos()).normalized())
            QGraphicsView.mouseMoveEvent(self, event)
        super(GraphicsView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in [QtCore.Qt.LeftButton, QtCore.Qt.RightButton]:
            self.changeRubberBand = False
            if self.rubberband.isVisible():
                self.rectChanged.emit(self.mapToScene(event.pos()).toPoint())
                self.rubberband.hide()

        super(GraphicsView, self).mouseReleaseEvent(event)

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

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()

if __name__ == '__main__':
    main()