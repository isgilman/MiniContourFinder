from tkinter import *
from tkinter.ttk import Frame, Button, Label, Style
from PIL import Image, ImageTk


class Example(Frame):

    def __init__(self):
        super().__init__()

        self.pack(fill=BOTH, expand=True)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)

        """Create image label"""
        path = "../Data/sandbox_data/dragonflywing.jpg"
        self.image = Image.open(path)
        self.__Winitial__, self.__Hinitial__ = self.image.size
        self.img_copy= self.image.copy()
        self.label_image = ImageTk.PhotoImage(self.image)
        self.label = Label(self, image=self.label_image)

        """Add image to grid and bind to resize"""
        self.label.grid(row=0, column=0, columnspan=2, rowspan=2, padx=5)
        self.label.bind('<Configure>', self._resize_image)

        """Add sliders"""
        self.k_blur_label = Label(self, text="k_blur").grid(row=0, column=3, pady=5, padx=5)
        self.k_blur_scale = Scale(self, from_=0, to=50, tickinterval=50, orient=HORIZONTAL, length=400)
        self.k_blur_scale.set(23)
        self.k_blur_scale.grid(row=0, column=3)

    def _resize_image(self, event):
        new_window_width = event.width
        new_window_height = event.height

        resize_ratio = min([new_window_height / float(self.__Hinitial__), new_window_width / float(self.__Winitial__)])
        new_width = int(self.__Winitial__ * resize_ratio)
        new_height = int(self.__Hinitial__ * resize_ratio)
        self.image = self.img_copy.resize((new_width, new_height))
        self.label_image = ImageTk.PhotoImage(self.image)
        self.label.configure(image=self.label_image)

def main():

    root = Tk()
    root.geometry("600x600")
    app = Example()
    root.mainloop()


if __name__ == '__main__':
    main()