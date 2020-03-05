from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPixmapCache
from circumscriptor import *
import pyperclip

class App(QWidget):
    def __init__(self, parent=None):
        super(App, self).__init__(parent=parent)
        ### Create window with grid layout ###
        self.resize(1000, 800)
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        self.setMouseTracking(True)

        ### Render image ###
        self.label = QLabel(self)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_path = sys.argv[1]
        self.cv2_image = cv2.imread(self.image_path)
        self.__originalH__, self.__originalW__, self.__originalD__ = self.cv2_image.shape

        # Get contours
        self.contours = mcf(image=self.cv2_image, skip_flood=True)
        self.contour_color = (255, 0, 0)
        self.highlight_color = (255, 0, 255)
        self.highlighted = []
        self.contours_clickable = False
        cv2.drawContours(self.cv2_image, contours=self.contours, contourIdx=-1, color=self.contour_color, thickness=2)
        if len(self.highlighted)>0:
            cv2.drawContours(self.cv2_image, contours=self.highlighted, contourIdx=-1, color=self.highlight_color, thickness=2)

        # Convert to Qimage object
        self.np_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)
        self.totalBytes = self.np_image.nbytes
        self.bytesPerLine = int(self.totalBytes / self.np_image.shape[0])
        self.qimage = QImage(self.np_image, self.np_image.shape[1], self.np_image.shape[0], self.bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(self.qimage)
        self.label.setPixmap(pixmap)
        grid_layout.addWidget(self.label, 0, 0, 12, 8) # Note that grid coords are (row, col, rowspan, colspan)
        self.label.mousePressEvent = self.get_image_pos
        self.label.setCursor(QtCore.Qt.CrossCursor)

        ### Create sliders ###
        self.k_blur = QSlider(QtCore.Qt.Horizontal)
        self.k_blur.setRange(2, 26)
        self.k_blur.setValue(5)
        self.k_blur.setTickInterval(1)
        self.k_blur.setTickPosition(QSlider.TicksBelow)
        self.blur_label = QLabel()
        self.blur_label.setText('k_blur: {}'.format(2*self.k_blur.value()-1))
        grid_layout.addWidget(self.blur_label, 0, 10)
        grid_layout.addWidget(self.k_blur, 0, 8, 1, 2)

        self.C = QSlider(QtCore.Qt.Horizontal)
        self.C.setRange(1, 15)
        self.C.setValue(3)
        self.C.setTickInterval(1)
        self.C.setTickPosition(QSlider.TicksBelow)
        self.C_label = QLabel()
        self.C_label.setText('C: {}'.format(self.C.value()))
        grid_layout.addWidget(self.C_label, 1, 10)
        grid_layout.addWidget(self.C, 1, 8, 1, 2)

        self.blocksize = QSlider(QtCore.Qt.Horizontal)
        self.blocksize.setRange(2, 100)
        self.blocksize.setValue(8)
        self.blocksize.setTickInterval(1)
        self.blocksize.setTickPosition(QSlider.TicksBelow)
        self.blocksize_label = QLabel()
        self.blocksize_label.setText('blocksize: {}'.format(2*self.blocksize.value()-1))
        grid_layout.addWidget(self.blocksize_label, 2, 10)
        grid_layout.addWidget(self.blocksize, 2, 8, 1, 2)

        self.k_laplacian = QSlider(QtCore.Qt.Horizontal)
        self.k_laplacian.setRange(2, 15)
        self.k_laplacian.setValue(3)
        self.k_laplacian.setTickInterval(1)
        self.k_laplacian.setTickPosition(QSlider.TicksBelow)
        self.laplacian_label = QLabel()
        self.laplacian_label.setText('k_laplacian: {}'.format(2*self.k_laplacian.value()-1))
        grid_layout.addWidget(self.laplacian_label, 3, 10)
        grid_layout.addWidget(self.k_laplacian, 3, 8, 1, 2)

        self.k_dilate = QSlider(QtCore.Qt.Horizontal)
        self.k_dilate.setRange(2, 52)
        self.k_dilate.setValue(3)
        self.k_dilate.setTickInterval(1)
        self.k_dilate.setTickPosition(QSlider.TicksBelow)
        self.dilate_label = QLabel()
        self.dilate_label.setText('k_dilate: {}'.format(2*self.k_dilate.value()-1))
        grid_layout.addWidget(self.dilate_label, 4, 10)
        grid_layout.addWidget(self.k_dilate, 4, 8, 1, 2)

        self.k_gradient = QSlider(QtCore.Qt.Horizontal)
        self.k_gradient.setRange(2, 52)
        self.k_gradient.setValue(2)
        self.k_gradient.setTickInterval(1)
        self.k_gradient.setTickPosition(QSlider.TicksBelow)
        self.gradient_label = QLabel()
        self.gradient_label.setText('k_gradient: {}'.format(2*self.k_gradient.value()-1))
        grid_layout.addWidget(self.gradient_label, 5, 10)
        grid_layout.addWidget(self.k_gradient, 5, 8, 1, 2)

        self.k_foreground = QSlider(QtCore.Qt.Horizontal)
        self.k_foreground.setRange(2, 15)
        self.k_foreground.setValue(4)
        self.k_foreground.setTickInterval(1)
        self.k_foreground.setTickPosition(QSlider.TicksBelow)
        self.foreground_label = QLabel()
        self.foreground_label.setText('k_foreground: {}'.format(2*self.k_foreground.value()-1))
        grid_layout.addWidget(self.foreground_label, 6, 10)
        grid_layout.addWidget(self.k_foreground, 6, 8, 1, 2)

        ### Set contour size ###
        self.Amin = QLineEdit()
        self.Amin.setPlaceholderText("Minimum contour area")
        self.setAmin = QPushButton("Enter")
        grid_layout.addWidget(self.Amin, 7, 8, 1, 2)
        grid_layout.addWidget(self.setAmin, 7, 10)

        self.Amax = QLineEdit()
        self.Amax.setPlaceholderText("Maximum contour area")
        self.setAmax = QPushButton("Enter")
        grid_layout.addWidget(self.Amax, 8, 8, 1, 2)
        grid_layout.addWidget(self.setAmax, 8, 10)

        ### Generate parameters for CLI ###
        self.CLIgenerator = QPushButton()
        self.CLIgenerator.setText("Copy CLI parameters to clipboard")
        grid_layout.addWidget(self.CLIgenerator, 9, 8, 1, 3)

        ### Text box ###
        self.box = QPlainTextEdit()
        self.box.insertPlainText("--k_blur 9 --C 3 --blocksize 15 --k_laplacian 5 --k_dilate 5 --k_gradient 3 --k_foreground 7 --Amin 50 --Amax 10e6")
        # grid_layout.addWidget(self.box, 11, 6, 1, 3)

        ### Reset parameters ###
        self.resetParams = QPushButton()
        self.resetParams.setText("Reset parameters")
        grid_layout.addWidget(self.resetParams, 10, 8, 1, 3)

        ### Color picker ###
        self.contour_colorPicker = QPushButton()
        self.contour_colorPicker.setText("Contour color")
        grid_layout.addWidget(self.contour_colorPicker, 11, 8, 1, 3)

        self.highlight_colorPicker = QPushButton()
        self.highlight_colorPicker.setText("Highlight color")
        grid_layout.addWidget(self.highlight_colorPicker, 12, 8, 1, 3)

        ### Contour selection ###
        self.select_contours = QCheckBox("Select contours")
        grid_layout.addWidget(self.select_contours, 13, 8, 1, 3)

        ### Connections ###
        # Sliders -> plots
        self.k_blur.valueChanged.connect(self.update_plot)
        self.C.valueChanged.connect(self.update_plot)
        self.blocksize.valueChanged.connect(self.update_plot)
        self.k_laplacian.valueChanged.connect(self.update_plot)
        self.k_dilate.valueChanged.connect(self.update_plot)
        self.k_gradient.valueChanged.connect(self.update_plot)
        self.k_foreground.valueChanged.connect(self.update_plot)
        # Sliders -> text
        self.k_blur.valueChanged.connect(self.update_text)
        self.C.valueChanged.connect(self.update_text)
        self.blocksize.valueChanged.connect(self.update_text)
        self.k_laplacian.valueChanged.connect(self.update_text)
        self.k_dilate.valueChanged.connect(self.update_text)
        self.k_gradient.valueChanged.connect(self.update_text)
        self.k_foreground.valueChanged.connect(self.update_text)
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
        self.contour_colorPicker.clicked.connect(self.update_plot)
        self.highlight_colorPicker.clicked.connect(self.update_highlight_color)
        self.highlight_colorPicker.clicked.connect(self.update_plot)

    def get_image_pos(self, event):
        if self.select_contours.isChecked():
            x = event.pos().x()
            y = event.pos().y()
            width_ratio = self.__originalW__/float(self.pixmap.width())
            height_ratio = self.__originalH__/float(self.pixmap.height())
            new_x = int(width_ratio*x)
            new_y = int(height_ratio*y)

            # Remove highlight contour if reselected
            in_highlighted = False
            if len(self.highlighted) > 0:
                for i, h in enumerate(self.highlighted):
                    if cv2.pointPolygonTest(contour=h, pt=(new_x, new_y), measureDist=False) == 1.0:
                        self.highlighted.pop(i)
                        in_highlighted = True
                        break
            # Try to add contour
            if not in_highlighted:
                for c in self.large_contours:
                    if cv2.pointPolygonTest(contour=c, pt=(new_x, new_y), measureDist=False) == 1.0:
                        self.highlighted.append(c)
                        break


        QPixmapCache.clear()
        self.cv2_image = cv2.imread(self.image_path)
        self.contours = mcf(image=self.cv2_image,
                            k_blur=2 * self.k_blur.value() - 1,
                            C=self.C.value(),
                            blocksize=2 * self.blocksize.value() - 1,
                            k_laplacian=2 * self.k_laplacian.value() - 1,
                            k_dilate=2 * self.k_dilate.value() - 1,
                            k_gradient=2 * self.k_gradient.value() - 1,
                            k_foreground=2 * self.k_foreground.value() - 1, skip_flood=True)
        try:
            Amin = int(self.Amin.text())
        except ValueError:
            Amin = int(1)
        try:
            Amax = int(self.Amax.text())
        except ValueError:
            Amax = int(self.__originalH__*self.__originalW__)

        self.large_contours = contour_size_selection(self.contours, Amin=Amin, Amax=Amax)
        cv2.drawContours(self.cv2_image, contours=self.large_contours, contourIdx=-1, color=self.contour_color, thickness=2)
        if len(self.highlighted) > 0:
            cv2.drawContours(self.cv2_image, contours=self.highlighted, contourIdx=-1, color=self.highlight_color, thickness=2)

        self.np_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(self.np_image, self.np_image.shape[1], self.np_image.shape[0], self.bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(self.qimage)
        self.pixmap = pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.width(), self.height())

    def copy_CLI(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self.box.toPlainText(), mode=cb.Clipboard)
        pyperclip.copy(self.box.toPlainText())

    def update_contour_color(self):
        self.contour_color = QColorDialog.getColor().getRgb()

    def update_highlight_color(self):
        self.highlight_color = QColorDialog.getColor().getRgb()

    def update_plot(self):
        QPixmapCache.clear()
        self.cv2_image = cv2.imread(self.image_path)
        self.contours = mcf(image=self.cv2_image,
                            k_blur=2 * self.k_blur.value() - 1,
                            C=self.C.value(),
                            blocksize=2 * self.blocksize.value() - 1,
                            k_laplacian=2 * self.k_laplacian.value() - 1,
                            k_dilate=2 * self.k_dilate.value() - 1,
                            k_gradient=2 * self.k_gradient.value() - 1,
                            k_foreground=2 * self.k_foreground.value() - 1, skip_flood=True)
        try:
            Amin = int(self.Amin.text())
        except ValueError:
            Amin = int(1)
        try:
            Amax = int(self.Amax.text())
        except ValueError:
            Amax = int(self.__originalH__*self.__originalW__)

        self.large_contours = contour_size_selection(self.contours, Amin=Amin, Amax=Amax)
        cv2.drawContours(self.cv2_image, contours=self.large_contours, contourIdx=-1, color=self.contour_color, thickness=2)
        if len(self.highlighted)>0:
            cv2.drawContours(self.cv2_image, contours=self.highlighted, contourIdx=-1, color=self.highlight_color, thickness=2)

        self.np_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(self.np_image, self.np_image.shape[1], self.np_image.shape[0], self.bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(self.qimage)
        self.pixmap = pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.width(), self.height())

    def update_text(self):
        self.blur_label.setText('k_blur: {}'.format(2*self.k_blur.value()-1))
        self.C_label.setText('C: {}'.format(self.C.value()))
        self.blocksize_label.setText('blocksize: {}'.format(2*self.blocksize.value()-1))
        self.laplacian_label.setText('k_laplacian: {}'.format(2*self.k_laplacian.value()-1))
        self.dilate_label.setText('k_dilate: {}'.format(2*self.k_dilate.value()-1))
        self.gradient_label.setText('k_gradient: {}'.format(2*self.k_gradient.value()-1))
        self.foreground_label.setText('k_foreground: {}'.format(2*self.k_foreground.value()-1))

    def on_CLIgenerator_clicked(self):
        try:
            Amin = int(self.Amin.text())
        except ValueError:
            Amin = int(1)
        try:
            Amax = int(self.Amax.text())
        except ValueError:
            Amax = int(10e6)
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
        self.Amin.clear()
        self.Amax.clear()

    def resizeEvent(self, event):
        self.cv2_image = cv2.imread(self.image_path)
        self.contours = mcf(image=self.cv2_image,
                       k_blur=2 * self.k_blur.value() - 1,
                       C=self.C.value(),
                       blocksize=2 * self.blocksize.value() - 1,
                       k_laplacian=2 * self.k_laplacian.value() - 1,
                       k_dilate=2 * self.k_dilate.value() - 1,
                       k_gradient=2 * self.k_gradient.value() - 1,
                       k_foreground=2 * self.k_foreground.value() - 1, skip_flood=True)
        try:
            Amin = int(self.Amin.text())
        except ValueError:
            Amin = int(1)
        try:
            Amax = int(self.Amax.text())
        except ValueError:
            Amax = int(self.__originalH__*self.__originalW__)

        self.large_contours = contour_size_selection(self.contours, Amin=Amin, Amax=Amax)
        cv2.drawContours(self.cv2_image, contours=self.large_contours, contourIdx=-1, color=self.contour_color, thickness=2)
        if len(self.highlighted)>0:
            cv2.drawContours(self.cv2_image, contours=self.highlighted, contourIdx=-1, color=self.highlight_color, thickness=2)

        self.np_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(self.np_image, self.np_image.shape[1], self.np_image.shape[0], self.bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(self.qimage)
        self.pixmap = pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)
        self.label.resize(self.width(), self.height())

if __name__ == '__main__':
    print("""
    ╭━━┳┳┳━┳━━┳┳┳━━┳━━┳┳┳╮
    ┃ ━┫┃╭━┫ ━┫┃┃┃┃┃  ┃┃┃┃
    ╰━━┻┻╯ ╰━━┻━┻┻┻╋━╮┣━┻╯
                   ╰━━╯
    """)

    app = QApplication(sys.argv)
    w = App()
    w.show()
    app.exec_()