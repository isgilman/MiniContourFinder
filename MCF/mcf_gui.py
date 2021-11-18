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

        # Initialize widget with layout
        widget = QWidget()
        layout = QVBoxLayout(widget)
        # Set contour app as central widget
        self.contourApp = ContourApp(self)
        layout.addWidget(self.contourApp)
        self.setCentralWidget(widget)
        self.contourApp.updatePlot()
        self.contourApp.viewer.updateView()

        '''Add toolbar'''
        toolbar = QToolBar()
        self.addToolBar(QtCore.Qt.LeftToolBarArea, toolbar)

        saveFile = QAction("&Save", self)
        saveFile.setShortcut("Ctrl+S")
        saveFile.setStatusTip("Save File")
        saveFile.triggered.connect(self._saveAll)
        toolbar.addAction(saveFile)

        openFile = QAction("&Open", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(self._fileOpen)
        toolbar.addAction(openFile)

        copyCLI = QAction("&Copy CLI command", self)
        copyCLI.setShortcut("Ctrl+C")
        copyCLI.setStatusTip('Copy CLI')
        copyCLI.triggered.connect(self._copyCLI)
        toolbar.addAction(copyCLI)

        contourColor = QAction("&Contour color", self)
        contourColor.setStatusTip('Change contour color')
        contourColor.triggered.connect(self._contourColorSelection)
        toolbar.addAction(contourColor)

        selectedColor = QAction("&Selected color", self)
        selectedColor.setStatusTip('Change selected contour color')
        selectedColor.triggered.connect(self._selectedColorSelection)
        toolbar.addAction(selectedColor)

        lineThickness = QAction("&Line thickness", self)
        lineThickness.setStatusTip('Line thickness')
        lineThickness.triggered.connect(self._setLineThickness)
        toolbar.addAction(lineThickness)

        detectScale = QAction("&Detect scale", self)
        detectScale.setStatusTip('Detect scale')
        detectScale.triggered.connect(self._detectScale)
        toolbar.addAction(detectScale)

        drawScaleBar = QAction("&Draw scale bar", self)
        drawScaleBar.setStatusTip('Draw scale')
        drawScaleBar.triggered.connect(self._drawScaleBar)
        toolbar.addAction(drawScaleBar)

        setScale = QAction("&Set scale", self)
        setScale.setStatusTip('Set scale')
        setScale.triggered.connect(self._setScale)
        toolbar.addAction(setScale)

        clearScale = QAction("&Clear scale", self)
        clearScale.setStatusTip('Clear scale')
        clearScale.triggered.connect(self._clearScale)
        toolbar.addAction(clearScale)

        resetParameters = QAction("&Reset parameters", self)
        resetParameters.setStatusTip('Reset parameters')
        resetParameters.triggered.connect(self._resetParameters)
        toolbar.addAction(resetParameters)

        selectAll = QAction("&Select all", self)
        selectAll.setStatusTip('Select all contours')
        selectAll.triggered.connect(self._selectAll)
        toolbar.addAction(selectAll)

        unselectAll = QAction("&Unselect all", self)
        unselectAll.setStatusTip('Unselect all contours')
        unselectAll.triggered.connect(self._unselectAll)
        toolbar.addAction(unselectAll)

        self.show()

    def _selectAll(self):
        if not self.contourApp.select_contours.isChecked():
            self.contourApp.select_contours.setChecked(True)
        
        self.contourApp.contourDF["selected"] = True
        self.contourApp.updatePlot()

    def _unselectAll(self):
        if not self.contourApp.select_contours.isChecked():
            self.contourApp.select_contours.setChecked(True)
        
        self.contourApp.contourDF["selected"] = False
        self.contourApp.updatePlot()


    def _resetParameters(self):
        self.contourApp.k_blur.setValue(5)
        self.contourApp.C.setValue(3)
        self.contourApp.blocksize.setValue(8)
        self.contourApp.k_laplacian.setValue(3)
        self.contourApp.k_dilate.setValue(3)
        self.contourApp.k_gradient.setValue(2)
        self.contourApp.k_foreground.setValue(4)
        self.contourApp.epsilon.setValue(3)
        self.contourApp.Amin.clear()
        self.contourApp.Amax.clear()
        self.contourApp.updatePlot()
        self.contourApp.updateText()

    def _drawScaleBar(self):
        if self.contourApp.select_contours.isChecked():
            print("Cannot draw scale bar while selecting contours")
            return
        else:
            self._clearScale()
            self.contourApp.drawing = True
            self.contourApp.scaleBar = []

    def _setScale(self):
        scaleDialog = TripleInputDialog()
        if self.contourApp.scaleBarPixelLength is not None:
            scaleDialog.first.setPlaceholderText(str(self.contourApp.scaleBarPixelLength))
        if self.contourApp.scaleBarUnitLength is not None:
            scaleDialog.second.setPlaceholderText(str(self.contourApp.scaleBarUnitLength))
        if self.contourApp.scaleBarUnits is not None:
            scaleDialog.third.setPlaceholderText(str(self.contourApp.scaleBarUnits))

        if scaleDialog.exec():
            try:
                self.contourApp.scaleBarPixelLength = int(eval(scaleDialog.getInputs()[0]))
                self.contourApp.scaleBarUnitLength = float(eval(scaleDialog.getInputs()[1]))
                self.contourApp.scaleBarUnits = scaleDialog.getInputs()[2]
            except ValueError:
                print("Invalid scale inputs")
                return

    def _detectScale(self):
        scaleBarInfo = get_scalebar_info(self.contourApp.cv2Image)
        if scaleBarInfo is not None:
            if len(scaleBarInfo) == 2:
                self.contourApp.scaleBar = scaleBarInfo[0]
                self.contourApp.scaleBarPixelLength = scaleBarInfo[1]
            elif len(scaleBarInfo) == 4:
                self.contourApp.scaleBar = scaleBarInfo[0]
                self.contourApp.scaleBarPixelLength = scaleBarInfo[1]
                self.contourApp.scaleBarUnitLength = scaleBarInfo[2]
                self.contourApp.scaleBarUnits = scaleBarInfo[3]
            else:
                self.contourApp.scaleBar = None
                print("Error reading scalebar. Try again or set manually.")
        else:
            self.contourApp.scaleBar = None
            
        self.contourApp.updatePlot()

    def _clearScale(self):
        self.contourApp.scaleBar = None
        self.contourApp.scaleBarPixelLength = None
        self.contourApp.scaleBarUnitLength = None
        self.contourApp.scaleBarUnits = None
        self.contourApp.drawing = False
        self.contourApp.updatePlot()

    def _setLineThickness(self):
        thicknessDialog, ok = QInputDialog.getText(self, "Set line thickness", "Line thickness")
        if ok:
            try:
                self.contourApp.contourThickness = int(eval(thicknessDialog))
                self.contourApp.updatePlot()
            except:
                print("Invalid thickness input")
                return
        else:
            return

    def _contourColorSelection(self):
        self.contourApp.contourColor = QColorDialog.getColor().getRgb()[:3][::-1]
        self.contourApp.updatePlot()

    def _selectedColorSelection(self):
        self.contourApp.highlightColor = QColorDialog.getColor().getRgb()[:3][::-1]
        self.contourApp.updatePlot()

    def _copyCLI(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        try: Amin = int(self.contourApp.Amin.text())
        except ValueError: Amin = int(1)
        try: Amax = int(self.contourApp.Amax.text())
        except ValueError: Amax = int(self.contourApp.cv2Image.shape[0]* self.contourApp.cv2Image.shape[1])

        CLItext = "--k_blur {} --C {} --blocksize {} --k_laplacian {} --k_dilate {} --k_gradient {} --k_foreground {} --Amin {} --Amax {}".format(
                2 * self.contourApp.k_blur.value() - 1,
                self.contourApp.C.value(),
                2 * self.contourApp.blocksize.value() - 1,
                2 * self.contourApp.k_laplacian.value() - 1,
                2 * self.contourApp.k_dilate.value() - 1,
                2 * self.contourApp.k_gradient.value() - 1,
                2 * self.contourApp.k_foreground.value() - 1,
                Amin, 
                Amax)
        cb.setText(CLItext, mode=cb.Clipboard)
        pyperclip.copy(CLItext)

    def _saveAll(self):
        saveDialog = QFileDialog.getSaveFileName(parent=self, 
        caption="File prefix",
        directory=Path(Path(sys.argv[1]).parent / Path(sys.argv[1]).stem).as_posix())

        if saveDialog[0] != '':
            if len(self.contourApp.contourDF) == 0 :
                exportDF = pd.DataFrame(np.zeros(shape=(len(self.contourApp.largeContours), 9)), 
                columns=["uuid4", "contour", "C", "kBlur", "blocksize", "kLaplacian","kDilate", "kGradient", "kForeground"], dtype=object)
                exportDF["contour"] = self.contourApp.largeContours
                exportDF["uuid4"] = [uuid.uuid4().hex for i in range(len(self.contourApp.largeContours))]
                exportDF["C"] = self.contourApp.C.value()
                exportDF["kBlur"] = self.contourApp.k_blur.value()
                exportDF["blocksize"] = self.contourApp.blocksize.value()
                exportDF["kLaplacian"] = self.contourApp.k_laplacian.value()
                exportDF["kDilate"] = self.contourApp.k_dilate.value()
                exportDF["kGradient"] = self.contourApp.k_gradient.value()
                exportDF["kForeground"] = self.contourApp.k_foreground.value()

                if self.use_approxPolys.isChecked():
                    currentContourDF["epsilon"] = self.epsilon.value()
                else:
                    currentContourDF["epsilon"] = None

            else:
                exportDF = self.contourApp.contourDF

            export_contour_data(image=sys.argv[1], contourDF=exportDF, conversion_factor=self.contourApp.unitsPerPixel, units=self.contourApp.scaleBarUnits,
                                output_dir=Path(saveDialog[0]).parent, prefix=Path(saveDialog[0]).stem)
            render_contour_plots(image=sys.argv[1], contours=exportDF["contour"].values, border_contour=None, contour_thickness=self.contourApp.contourThickness,
                                output_dir=Path(saveDialog[0]).parent, prefix=Path(saveDialog[0]).stem, color=self.contourApp.highlightColor)
        else: return

    def _fileOpen(self):
        fileFilter = 'Contour data file (*.json)'
        openDialog = QFileDialog.getOpenFileName(self, 
            caption = "Open contour data file (.json)",
            directory = Path(Path(sys.argv[1]).parent / Path(sys.argv[1]).stem).as_posix(),
            filter = fileFilter)

        if openDialog[0] != '':
            self.contourApp.select_contours.setChecked(True)
            loaded = pd.read_json(openDialog[0])
            loaded["selected"] = True
            loaded = loaded[self.contourApp.contourDF.columns]
            loaded["contour"] = loaded["contour"].apply(np.array)
            print("[{}] Loaded {} contours from {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), len(loaded), openDialog[0]))
            self.contourApp.contourDF = self.contourApp.contourDF.append(loaded)
            self.contourApp.updatePlot()
        else:return

    def _closeEvent(self, event):
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

        ''' Create window with grid layout '''
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.click_x, self.click_y = (0,0)
        self.setMouseTracking(True)

        ''' Initialize viewer '''
        self.viewer = GraphicsView(self)
        self.grid_layout.addWidget(self.viewer, 0, 0, -1, 8) # Note that grid coords are (row, col, rowspan, colspan)
        self.viewer.updateView()
        self.viewer.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.viewer.setAlignment(QtCore.Qt.AlignTop)

        ''' Add image and contours '''
        self.image_path = Path(sys.argv[1])
        image = cv2.imread(self.image_path.as_posix())
        if image is None:
            image = vector2cv2(self.image_path.as_posix())

        wdir = self.image_path.parent
        results = wdir.glob("{}.denoise.*".format(self.image_path.stem))
        try:
            self.denoised_path = next(results)
            self.cv2Image = cv2.imread(self.denoised_path.as_posix())
            print("[{}] Found existing denoised file ({})".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), self.denoised_path.as_posix()))
        except StopIteration:
            print("[{}] Denoising image...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
            self.cv2Image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            self.denoised_path = Path("{}/{}.denoise{}".format(wdir.as_posix(), self.image_path.stem, ".png"))
            cv2.imwrite(filename=self.denoised_path.as_posix(), img=self.cv2Image)

            print("[{}] Created temporary denoised file ({})".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), self.denoised_path.as_posix()))

        # Get contours
        self.contours = mcf(image=self.cv2Image)
        self.contourColor = (255, 0, 0)
        self.highlightColor = (255, 0, 255)
        self.contourThickness = 3
        self.contourDF = pd.DataFrame(columns=["uuid4", "contour", "C", "kBlur", "blocksize", "kLaplacian", "kDilate", "kGradient", "kForeground","epsilon", "selected"], dtype=object)
        cv2.drawContours(self.cv2Image, contours=self.contours, contourIdx=-1, color=self.contourColor, thickness=self.contourThickness)
        if len(self.contourDF)>0:
            cv2.drawContours(self.cv2Image, contours=self.contourDF["contour"].values, contourIdx=-1, color=self.highlightColor, thickness=self.contourThickness)

        # Convert to Qimage object
        self.pixmap = cv2pixmap(self.cv2Image)
        self.viewer.setPhoto(self.pixmap)
        self.viewer.updateView()

        ''' Scale bar info '''
        self.scaleBar = None
        self.scaleBarUnits = None
        self.scaleBarPixelLength = None
        self.scaleBarUnitLength = None
        self.unitsPerPixel = None
        self.drawing = False

        ''' Create sliders'''
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

        ''' Set contour size '''
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

        ''' Contour selection '''
        self.select_contours = QCheckBox("Select contours")
        self.grid_layout.addWidget(self.select_contours, 9, 10, 1, 1)
        self.select_contours.stateChanged.connect(self.selectContoursChecked)

        ''' Approximate polygons '''
        self.use_approxPolys = QCheckBox("Approximate polygons (epsilon)")
        self.grid_layout.addWidget(self.use_approxPolys, 9, 8, 1, 1)
        self.epsilon = QSlider(QtCore.Qt.Horizontal)
        self.epsilon.setRange(1, 20)
        self.epsilon.setValue(1)
        self.epsilon.setTickInterval(1)
        self.epsilon.setTickPosition(QSlider.TicksBelow)
        self.epsilon_label = QLabel()
        self.epsilon_label.setText('episilon: {}'.format(self.epsilon.value()))
        self.grid_layout.addWidget(self.epsilon_label, 10, 10)
        self.grid_layout.addWidget(self.epsilon, 10, 8, 1, 2)

        ''' Convex hulls '''
        self.use_convexHulls = QCheckBox("Use convex hulls")
        self.grid_layout.addWidget(self.use_convexHulls, 9, 9, 1, 1)

        ''' Connections '''
        # Sliders -> plots
        self.k_blur.valueChanged.connect(self.updatePlot)
        self.C.valueChanged.connect(self.updatePlot)
        self.blocksize.valueChanged.connect(self.updatePlot)
        self.k_laplacian.valueChanged.connect(self.updatePlot)
        self.k_dilate.valueChanged.connect(self.updatePlot)
        self.k_gradient.valueChanged.connect(self.updatePlot)
        self.k_foreground.valueChanged.connect(self.updatePlot)
        self.epsilon.valueChanged.connect(self.updatePlot)
        
        # Sliders -> text
        self.k_blur.valueChanged.connect(self.updateText)
        self.C.valueChanged.connect(self.updateText)
        self.blocksize.valueChanged.connect(self.updateText)
        self.k_laplacian.valueChanged.connect(self.updateText)
        self.k_dilate.valueChanged.connect(self.updateText)
        self.k_gradient.valueChanged.connect(self.updateText)
        self.k_foreground.valueChanged.connect(self.updateText)
        self.epsilon.valueChanged.connect(self.updateText)
        
        # Line edits -> plots
        self.Amin.returnPressed.connect(self.updatePlot)
        self.setAmin.clicked.connect(self.updatePlot)
        self.Amax.returnPressed.connect(self.updatePlot)
        self.setAmax.clicked.connect(self.updatePlot)

        # Connect photo click and rubberband
        self.viewer.photoClicked.connect(self.photoClicked)
        self.viewer.rectChanged.connect(self.rectChange)
        self.updatePlot()

        self.sliders = [self.k_blur, self.C, self.blocksize, self.k_laplacian, self.k_dilate, self.k_gradient, self.k_foreground, self.epsilon]

    def photoClicked(self, pos):
        if self.select_contours.isChecked():
            self.click_x, self.click_y = (pos.x(), pos.y())

            if self.viewer.button == 1: # Click addition
                for i, row in self.contourDF[self.contourDF["selected"]==False].iterrows():
                    if cv2.pointPolygonTest(contour=row["contour"], pt=(self.click_x, self.click_y), measureDist=False) == 1.0:
                        self.contourDF.loc[i, "selected"] = True
                        break

            elif self.viewer.button == 2: # Click subtraction
                if self.contourDF["selected"].any():
                    for i, row in self.contourDF[self.contourDF["selected"]==True].iterrows():
                        if cv2.pointPolygonTest(contour=row["contour"], pt=(self.click_x, self.click_y), measureDist=False) == 1.0:
                            self.contourDF.loc[i, "selected"] = False
                            break

            else: return
            self.updatePlot()

        elif self.drawing:
            if len(self.scaleBar) == 0:
                self.scaleBar.append(pos.x())
                self.scaleBar.append(pos.y())
                self.scaleBarPixelLength = None
                self.updatePlot()
                
            elif len(self.scaleBar) == 2:
                self.scaleBar.append(pos.x())
                self.scaleBar.append(pos.y())
                self.scaleBarPixelLength = np.sqrt((self.scaleBar[2]-self.scaleBar[0])**2+(self.scaleBar[3]-self.scaleBar[1])**2)
                self.updatePlot()

                scaleDialog = TripleInputDialog()
                scaleDialog.first.setText(str(self.scaleBarPixelLength))

                if scaleDialog.exec():
                    try:
                        self.scaleBarPixelLength = int(eval(scaleDialog.getInputs()[0]))
                        self.scaleBarUnitLength = float(eval(scaleDialog.getInputs()[1]))
                        self.scaleBarUnits = scaleDialog.getInputs()[2]
                    except:
                        print("Invalid scale inputs") 
                        self.scaleBar = None
                        self.scaleBarPixelLength = None
                        self.scaleBarUnitLength = None
                        self.scaleBarUnits = None
                self.drawing = False

            else:
                self.scaleBar = None
                self.scaleBarPixelLength = None
                self.scaleBarUnitLength = None
                self.scaleBarUnits = None
                self.drawing = False

            self.updatePlot()

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
                self.contourDF = RectangleOverlapTest(image=self.cv2Image, contourDF=self.contourDF,
                                                x=min(xi, xf), y=min(yi, yf), width=w, height=h)

            elif self.viewer.button == 2: # Rubberband subtraction
                self.contourDF = RectangleOverlapTest(image=self.cv2Image, contourDF=self.contourDF,
                                                        x=min(xi, xf), y=min(yi, yf), width=w, height=h, REMOVE=True)

            else: return
            
            self.updatePlot()

    def selectContoursChecked(self):
        if self.select_contours.isChecked():
            self.viewer.selectContoursChecked = True
            for slider in self.sliders:
                slider.setEnabled(False)

            self.use_approxPolys.setEnabled(False)
            self.use_convexHulls.setEnabled(False)

            # Create a temporary selection criterion
            self.contourDF["selected"] = True
            currentContourDF = pd.DataFrame(columns=["uuid4", "contour", "C", "kBlur", "blocksize", "kLaplacian", "kDilate", "kGradient", "kForeground"], dtype=object)
            currentContourDF["contour"] = self.largeContours
            currentContourDF["uuid4"] = [uuid.uuid4().hex for i in range(len(self.largeContours))]
            currentContourDF["C"] = self.C.value()
            currentContourDF["kBlur"] = self.k_blur.value()
            currentContourDF["blocksize"] = self.blocksize.value()
            currentContourDF["kLaplacian"] = self.k_laplacian.value()
            currentContourDF["kDilate"] = self.k_dilate.value()
            currentContourDF["kGradient"] = self.k_gradient.value()
            currentContourDF["kForeground"] = self.k_foreground.value()
            currentContourDF["selected"] = False
    
            if self.use_approxPolys.isChecked():
                currentContourDF["epsilon"] = self.epsilon.value()
            else:
                currentContourDF["epsilon"] = None

            self.contourDF = self.contourDF.append(currentContourDF).reset_index(drop=True)

        else:
            self.viewer.selectContoursChecked = False
            for slider in self.sliders:
                slider.setEnabled(True)
            self.use_approxPolys.setEnabled(True)
            self.use_convexHulls.setEnabled(True)
            
            # Retain only selected contours
            self.contourDF = self.contourDF[self.contourDF["selected"]==True].reset_index(drop=True)


    def resizeEvent(self, event):
        self.updatePlot()

    def updatePlot(self):
        # Clear image
        QPixmapCache.clear()
        self.cv2Image = cv2.imread(self.denoised_path.as_posix())
        # If actively tuning contours...
        if not self.select_contours.isChecked():
            self.contours = mcf(image=self.cv2Image,
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
            except ValueError: Amax = int(self.cv2Image.shape[0]* self.cv2Image.shape[1])
            # Refine contours
            self.largeContours = contour_size_selection(self.contours, Amin=Amin, Amax=Amax)
            if self.use_approxPolys.isChecked():
                self.largeContours = [cv2.approxPolyDP(curve=c, epsilon=self.epsilon.value(), closed=True) for c in self.largeContours]
            if self.use_convexHulls.isChecked():
                self.largeContours = [cv2.convexHull(c) for c in self.largeContours]

        else:
            self.largeContours = self.contourDF["contour"].values

        cv2.drawContours(self.cv2Image, contours=self.largeContours, contourIdx=-1, color=self.contourColor, thickness=self.contourThickness)
        if self.contourDF["selected"].any():
            cv2.drawContours(self.cv2Image, contours=self.contourDF[self.contourDF["selected"]==True]["contour"].values, contourIdx=-1, color=self.highlightColor, thickness=self.contourThickness)

        # Add scale bar
        if self.scaleBar is not None:
            if self.drawing:
                cv2.circle(self.cv2Image, (self.scaleBar[0], self.scaleBar[1]), radius=5, color=(0,255,0), thickness=-1)
                if len(self.scaleBar) == 4:
                    cv2.circle(self.cv2Image, (self.scaleBar[2], self.scaleBar[3]), radius=5, color=(0,255,0), thickness=-1)
                    cv2.line(self.cv2Image, (self.scaleBar[0], self.scaleBar[1]), (self.scaleBar[2], self.scaleBar[3]), (0,255,0),5)
            else:
                cv2.line(self.cv2Image, (self.scaleBar[0], self.scaleBar[1]), (self.scaleBar[2], self.scaleBar[3]), (0,255,0),5)
       
        # Convert to QImage
        self.pixmap = cv2pixmap(self.cv2Image)
        self.viewer.setPhoto(self.pixmap)
        self.viewer.updateView()


    def updateText(self):
        self.blur_label.setText('k_blur: {}'.format(2*self.k_blur.value()-1))
        self.C_label.setText('C: {}'.format(self.C.value()))
        self.blocksize_label.setText('blocksize: {}'.format(2*self.blocksize.value()-1))
        self.laplacian_label.setText('k_laplacian: {}'.format(2*self.k_laplacian.value()-1))
        self.dilate_label.setText('k_dilate: {}'.format(2*self.k_dilate.value()-1))
        self.gradient_label.setText('k_gradient: {}'.format(2*self.k_gradient.value()-1))
        self.foreground_label.setText('k_foreground: {}'.format(2*self.k_foreground.value()-1))
        self.epsilon_label.setText('epsilon: {}'.format(self.epsilon.value()))

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