from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QPixmapCache
from circumscriptor import *
import pyperclip
import pathlib as pl
import matplotlib
matplotlib.use('TkAgg')

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.contour_app = ContourApp(self)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.contour_app)
        self.setCentralWidget(widget)

        toolbar = QToolBar()
        self.addToolBar(toolbar)

        saveFile = QAction("&Save", self)
        saveFile.setShortcut("Ctrl+S")
        saveFile.setStatusTip('Save File')
        saveFile.triggered.connect(self.file_save)
        toolbar.addAction(saveFile)

        openFile = QAction("&Open", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(self.file_open)
        toolbar.addAction(openFile)

        self.setStatusBar(QStatusBar(self))
        self.resize(1000, 600)
        self.setWindowTitle('circumGUI - {}'.format(sys.argv[1]))
        self.show()

    def file_save(self):
        save_info = QFileDialog.getSaveFileName(self, "Save location")
        output_path = pl.Path(save_info[0])
        output_stem = output_path.stem
        output_dir = output_path.parent
        if len(self.contour_app.highlighted) < 1:
            export_contours = self.contour_app.large_contours
        else:
            export_contours = self.contour_app.highlighted

        export_contour_data(image=sys.argv[1], contours=export_contours,
                            output_dir=output_dir, prefix=output_stem)
        render_contour_plots(image=sys.argv[1], contours=export_contours, border_contour=None, contour_thickness=self.contour_app.contour_thickness.value(),
                             output_dir=output_dir, prefix=output_stem, color=self.contour_app.highlight_color)

    def file_open(self):
        open_info = QFileDialog.getOpenFileName(self, "Open contour file (.pkl)", "./")
        if open_info != ('', ''):
            pickle = pd.read_pickle(open_info[0])
            print("[{}] Loaded contours from {}".format(datetime.now().strftime('%d %b %Y %H:%M:%S'), open_info[0]))
            self.contour_app.select_contours.setChecked(True)
            self.contour_app.contours = list(pickle["contour"])
            self.contour_app.large_contours = list(pickle["contour"])
            self.contour_app.highlighted = list(pickle["contour"])
            self.contour_app.update_plot()

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
    def __init__(self, parent=None):
        super(ContourApp, self).__init__(parent=parent)

        ### Create window with grid layout ###
        self.resize(1000, 800)
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.setMouseTracking(True)
        self.click_x, self.click_y = (0, 0)
        self.release_x, self.release_y = (0, 0)

        ### Render image ###
        self.viewer = PhotoViewer(self)
        self.grid_layout.addWidget(self.viewer, 0, 0, -1, 8) # Note that grid coords are (row, col, rowspan, colspan)
        self.viewer.fitInView()
        self.viewer.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.viewer.setAlignment(QtCore.Qt.AlignTop)
        self.image_path = sys.argv[1]
        """Denoise"""
        print("[{}] Denoising image...".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
        self.cv2_image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        # self.cv2_image = cv2.fastNlMeansDenoisingColored(self.cv2_image.copy(), None, 10, 10, 7, 21)
        print("[{}] Finished denoising".format(datetime.now().strftime('%d %b %Y %H:%M:%S')))
        self.__originalH__, self.__originalW__, self.__originalD__ = self.cv2_image.shape

        # Get contours
        self.contours = mcf(image=self.cv2_image, skip_flood=True)
        self.contour_color = (255, 0, 0)
        self.highlight_color = (255, 0, 255)
        self.highlighted = []
        cv2.drawContours(self.cv2_image, contours=self.contours, contourIdx=-1, color=self.contour_color, thickness=3)
        if len(self.highlighted)>0:
            cv2.drawContours(self.cv2_image, contours=self.highlighted, contourIdx=-1, color=self.highlight_color, thickness=3)

        # Convert to Qimage object
        self.np_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)
        self.totalBytes = self.np_image.nbytes
        self.bytesPerLine = int(self.totalBytes / self.np_image.shape[0])
        self.qimage = QImage(self.np_image, self.np_image.shape[1], self.np_image.shape[0], self.bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap(self.qimage)

        self.viewer.setPhoto(self.pixmap)
        self.viewer.fitInView()
        # Set up rubberband selection
        self.rubberband = QRubberBand(QRubberBand.Rectangle, self.viewer)

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
        self.blocksize.setRange(2, 100)
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

        ### Generate parameters for CLI ###
        self.CLIgenerator = QPushButton()
        self.CLIgenerator.setText("Copy CLI parameters to clipboard")
        self.grid_layout.addWidget(self.CLIgenerator, 9, 8, 1, 3)

        ### Text box ###
        self.box = QPlainTextEdit()
        self.box.insertPlainText("--k_blur 9 --C 3 --blocksize 15 --k_laplacian 5 --k_dilate 5 --k_gradient 3 --k_foreground 7 --Amin 50 --Amax 10e6")

        ### Reset parameters ###
        self.resetParams = QPushButton()
        self.resetParams.setText("Reset parameters")
        self.grid_layout.addWidget(self.resetParams, 10, 8, 1, 3)

        ### Color picker ###
        self.contour_colorPicker = QPushButton()
        self.contour_colorPicker.setText("Contour color")
        self.grid_layout.addWidget(self.contour_colorPicker, 11, 8, 1, 3)

        self.highlight_colorPicker = QPushButton()
        self.highlight_colorPicker.setText("Highlight color")
        self.grid_layout.addWidget(self.highlight_colorPicker, 12, 8, 1, 3)

        ### Contour thickness ###
        self.contour_thickness = QSlider(QtCore.Qt.Horizontal)
        self.contour_thickness.setRange(1, 10)
        self.contour_thickness.setValue(3)
        self.contour_thickness.setTickInterval(1)
        self.contour_thickness.setTickPosition(QSlider.TicksBelow)
        self.thickness_label = QLabel()
        self.thickness_label.setText('Contour thickness: {}'.format(self.contour_thickness.value()))
        self.grid_layout.addWidget(self.thickness_label, 13, 10)
        self.grid_layout.addWidget(self.contour_thickness, 13, 8, 1, 2)

        ### Contour selection ###
        self.select_contours = QCheckBox("Select contours")
        self.grid_layout.addWidget(self.select_contours, 14, 8, 1, 1)
        self.select_contours.stateChanged.connect(self.selectContoursChecked)

        ### Approximate polygons ###
        self.use_approxPolys = QCheckBox("Use approximate polygons")
        self.grid_layout.addWidget(self.use_approxPolys, 14, 9, 1, 1)

        self.epsilon = QSlider(QtCore.Qt.Horizontal)
        self.epsilon.setRange(1, 50)
        self.epsilon.setValue(3)
        self.epsilon.setTickInterval(1)
        self.epsilon.setTickPosition(QSlider.TicksBelow)
        self.epsilon_label = QLabel()
        self.epsilon_label.setText('episilon: {}'.format(self.epsilon.value()))
        self.grid_layout.addWidget(self.epsilon_label, 15, 10)
        self.grid_layout.addWidget(self.epsilon, 15, 8, 1, 2)

        ### Convex hulls ###
        self.use_convexHulls = QCheckBox("Use convex hulls")
        self.grid_layout.addWidget(self.use_convexHulls, 14, 10, 1, 1)

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

    def mousePressEvent(self, event):
        if self.viewer._photo.isUnderMouse():
            self.origin = self.viewer.mapToScene(event.pos()).toPoint()
            self.rubberband.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
            self.rubberband.show()

            self.click_x = self.viewer.mapToScene(event.pos()).toPoint().x()
            self.click_y = self.viewer.mapToScene(event.pos()).toPoint().y()
            print(self.viewer.width(), self.viewer.height())
            print("Viewer: {}, {} ".format(self.click_x, self.click_y))
            print("App: {}, {}".format(event.pos().x(), event.pos().y()))

            if self.select_contours.isChecked():
                # Remove contours with right click
                if event.button() == QtCore.Qt.RightButton:
                    if len(self.highlighted) > 0:
                        for i, h in enumerate(self.highlighted):
                            if cv2.pointPolygonTest(contour=h, pt=(self.click_x, self.click_y), measureDist=False) == 1.0:
                                self.highlighted.pop(i)
                                break
                # Add contour with left click
                if event.button() == QtCore.Qt.LeftButton:
                    for c in self.large_contours:
                        if cv2.pointPolygonTest(contour=c, pt=(self.click_x, self.click_y), measureDist=False) == 1.0:
                            self.highlighted.append(c)
                            break
                self.update_plot()

    def mouseMoveEvent(self, event):
        if self.rubberband.isVisible():
            self.rubberband.setGeometry(QtCore.QRect(self.origin, self.viewer.mapToScene(event.pos()).toPoint()).normalized() & self.qimage.rect())

    def mouseReleaseEvent(self, event):
        self.release = self.viewer.mapToScene(event.pos()).toPoint()
        self.release_x = self.viewer.mapToScene(event.pos()).toPoint().x()
        self.release_y = self.viewer.mapToScene(event.pos()).toPoint().y()

        if self.rubberband.isVisible():
            self.rubberband.hide()
        if self.select_contours.isChecked():
            width = abs(self.release_x-self.click_x)
            height = abs(self.release_y-self.click_y)
            if self.origin != self.release:
                if event.button() == QtCore.Qt.LeftButton:
                    selected = RectangleOverlapTest(image=self.cv2_image, contours=self.large_contours,
                                                    x=min(self.click_x, self.release_x),
                                                    y=min(self.click_y, self.release_y),
                                                    width=width, height=height)
                    new_highlighted = ContourOverlapTest(self.cv2_image, selected, self.highlighted, return_overlapping=False)
                    self.highlighted += new_highlighted

                if event.button() == QtCore.Qt.RightButton:
                    self.highlighted = RectangleOverlapTest(image=self.cv2_image, contours=self.highlighted,
                                                            x=min(self.click_x, self.release_x),
                                                            y=min(self.click_y, self.release_y),
                                                            width=width, height=height, REMOVE=True)
                self.update_plot()

    def copy_CLI(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self.box.toPlainText(), mode=cb.Clipboard)
        pyperclip.copy(self.box.toPlainText())

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

    def resizeEvent(self, event):
        self.update_plot()

    def update_plot(self):
        QPixmapCache.clear()
        # Get contours
        self.cv2_image = cv2.imread(self.image_path)
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
            except ValueError: Amax = int(self.__originalH__ * self.__originalW__)
            # Refine contours
            self.large_contours = contour_size_selection(self.contours, Amin=Amin, Amax=Amax)
            if self.use_approxPolys.isChecked():
                self.large_contours = [cv2.approxPolyDP(curve=c, epsilon=self.epsilon.value(), closed=True) for c in self.large_contours]
            if self.use_convexHulls.isChecked():
                self.large_contours = [cv2.convexHull(c) for c in self.large_contours]

        cv2.drawContours(self.cv2_image, contours=self.large_contours, contourIdx=-1, color=self.contour_color, thickness=self.contour_thickness.value())
        if len(self.highlighted) > 0:
            cv2.drawContours(self.cv2_image, contours=self.highlighted, contourIdx=-1, color=self.highlight_color, thickness=self.contour_thickness.value())
        cv2.circle(self.cv2_image, (self.click_x, self.click_y), 10, (0, 0, 255), -1)


        # Convert to QImage
        self.np_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(self.np_image, self.np_image.shape[1], self.np_image.shape[0], self.bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap(self.qimage)
        self.viewer.setPhoto(self.pixmap)

class PhotoViewer(QtWidgets.QGraphicsView):

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
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

        self.click_x = 0
        self.click_y = 0

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                self.factor = min(viewrect.width() / scenerect.width(),
                                 viewrect.height() / scenerect.height())
                self.scale(self.factor, self.factor)

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
            self.fitInView()
            zoom_key = True
        if zoom_key:
            if self._zoom > 0:
                self.scale(self.factor, self.factor)
            elif self._zoom == 0:
                self.fitInView()

if __name__ == '__main__':
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