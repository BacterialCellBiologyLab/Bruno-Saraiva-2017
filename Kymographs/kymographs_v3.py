import numpy as np
import tkMessageBox
import Tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from collections import OrderedDict
from skimage.io import imread_collection, imsave
from skimage.color import rgb2gray, gray2rgb
from skimage.draw import line
from skimage.exposure import rescale_intensity
from skimage.util import img_as_float
from tkFileDialog import askopenfilename, asksaveasfilename


class Interface(object):
    """Test"""

    def __init__(self):
        self.kymogen = KymoGen()
        self.image_index = 1
        self.len_img_stack = 0
        self.overlay = False
        self.images_loaded = False

        self.store_datapoints = []

        self.cid = None

        self.gui = tk.Tk()
        self.gui.wm_title("KymoGen")

        self.top_frame = tk.Frame(self.gui, width=1200, height=10)
        self.top_frame.pack(fill="x")

        self.load_stack_button = tk.Button(self.top_frame,
                                           command=self.load_stack,
                                           text="Load Stack",
                                           width=10)
        self.load_stack_button.pack(side="left")

        self.draw_button = tk.Button(self.top_frame,
                                     command=self.draw_region_of_interest,
                                     text="Draw ROI",
                                           width=10)
        self.draw_button.pack(side="left")
        self.draw_button.config(state="disabled")

        self.next_image_button = tk.Button(self.top_frame,
                                           command=self.next_image,
                                           text="Next Image",
                                           width=10)
        self.next_image_button.pack(side="right")
        self.next_image_button.config(state="disabled")

        self.previous_image_button = tk.Button(self.top_frame,
                                               command=self.previous_image,
                                               text="Previous Image",
                                           width=10)
        self.previous_image_button.pack(side="right")
        self.previous_image_button.config(state="disabled")

        self.image_index_status = tk.StringVar()
        self.image_index_status.set("No Image Loaded")

        self.image_index_label = tk.Label(self.top_frame,
                                          textvariable=self.image_index_status)
        self.image_index_label.pack(side="right")

        self.show_overlay_button = tk.Button(self.top_frame,
                                             command=self.show_overlay,
                                             text="Show Overlay",
                                             width=10)
        self.show_overlay_button.pack(side="left")
        self.show_overlay_button.config(state="disabled")

        self.hide_overlay_button = tk.Button(self.top_frame,
                                             command=self.hide_overlay,
                                             text="Hide Overlay",
                                             width=10)
        self.hide_overlay_button.pack(side="left")
        self.hide_overlay_button.config(state="disabled")

        self.gen_kymo_button = tk.Button(self.top_frame,
                                         command=self.gen_kymo,
                                         text="Generate Kymograph")
        self.gen_kymo_button.pack(side="left")
        self.gen_kymo_button.config(state="disabled")

        self.gen_kymo_button_3px_av = tk.Button(self.top_frame,
                                                command=self.gen_kymo_3px_average,
                                                text="Generate Kymograph 3px Average")
        self.gen_kymo_button_3px_av.pack(side="left")
        self.gen_kymo_button_3px_av.config(state="disabled")

        self.rescaling_label = tk.Label(self.top_frame, text="Use Rescaling: ")
        self.rescaling_label.pack(side="left")

        self.rescaling_value = tk.BooleanVar()
        self.rescaling_label_checkbox = tk.Checkbutton(self.top_frame, variable=self.rescaling_value,
                                                       onvalue=True, offvalue=False)
        self.rescaling_label_checkbox.pack(side="left")
        self.rescaling_value.set(False)

        self.middle_frame = tk.Frame(self.gui)
        self.middle_frame.pack(fill="x")

        self.fig = plt.figure(figsize=(22, 11), frameon=True)
        self.canvas = FigureCanvasTkAgg(self.fig, self.middle_frame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side="top")

        self.ax = plt.subplot(111)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self.ax.axis("off")
        plt.autoscale(True)
        self.ax.format_coord = self.remove_coord

        self.canvas.show()
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.middle_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill="both")

    def show_overlay(self):
        self.overlay = True

        self.show_image()

    def hide_overlay(self):
        self.overlay = False

        self.show_image()

    def gen_kymo(self):
        self.kymogen.rescaling = self.rescaling_value.get()
        self.kymogen.generate_kymograph()

    def gen_kymo_3px_average(self):
        self.kymogen.rescaling = self.rescaling_value.get()
        self.kymogen.generate_kymograph_3px_average()

    def load_stack(self):
        self.kymogen.img_stack = imread_collection(askopenfilename())
        self.len_img_stack = len(self.kymogen.img_stack)
        self.draw_button.config(state="active")
        self.next_image_button.config(state="active")
        self.image_index_status.set(str(self.image_index) + " of " + str(self.len_img_stack))

        self.show_image()

    def next_image(self):
        self.image_index += 1
        self.previous_image_button.config(state="active")

        if self.image_index == self.len_img_stack:
            self.next_image_button.config(state="disabled")

        self.image_index_status.set(str(self.image_index) + " of " + str(self.len_img_stack))
        self.show_image()

    def previous_image(self):
        self.image_index -= 1
        self.next_image_button.config(state="active")

        if self.image_index == 1:
            self.previous_image_button.config(state="disabled")

        self.image_index_status.set(str(self.image_index) + " of " + str(self.len_img_stack))
        self.show_image()

    def show_image(self):
        if self.images_loaded is True:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.cla()
            self.ax.axis("off")

            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        else:
            self.images_loaded = True
            self.ax.cla()
            self.ax.axis("off")

        if self.overlay is False:
            image = self.kymogen.img_stack[self.image_index-1]
        else:
            image = self.kymogen.img_stack_w_roi[self.image_index-1]

        image = img_as_float(image)
        #image = rescale_intensity(image)

        self.ax.imshow(image)

        self.canvas.draw()
        self.canvas.show()

    def overlay_roi(self):
        self.kymogen.img_stack_w_roi = []
        for img in self.kymogen.img_stack:
            img = gray2rgb(img)

            for coords in self.kymogen.roi:
                img[coords[0], coords[1]] = (255, 0, 255)

            self.kymogen.img_stack_w_roi.append(img)
        self.overlay = True

        self.show_overlay_button.config(state="active")
        self.hide_overlay_button.config(state="active")
        self.gen_kymo_button.config(state="active")
        self.gen_kymo_button_3px_av.config(state="active")
        self.show_image()

    def stop_points(self, event):
        if event.button == 3:
            self.canvas.mpl_disconnect(self.cid_press)
            self.canvas.mpl_disconnect(self.cid_motion)
            self.canvas.mpl_disconnect(self.cid_release)

            self.kymogen.roi = np.array(list(OrderedDict.fromkeys(self.store_datapoints)))
            x0, y0 = np.amin(self.kymogen.roi, axis=0)
            x1, y1 = np.amax(self.kymogen.roi, axis=0)
            self.kymogen.box = x0, y0, x1, y1
            self.store_datapoints = []

            self.overlay_roi()

    def start_points(self, event):

        if event.button == 3:
            self.cid_motion = self.canvas.mpl_connect("motion_notify_event",
                                                      self.get_points_of_roi)

    def get_points_of_roi(self, event):
        self.store_datapoints.append((int(event.ydata), int(event.xdata)))

    def draw_region_of_interest(self):
        self.cid_press = self.canvas.mpl_connect("button_press_event", self.start_points)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self.stop_points)

    def remove_coord(self, x, y):
        """"Hack" to remove the mpl coordinates"""
        return str(x) + " , " + str(y)


class KymoGen(object):
    def __init__(self):
        self.img_stack = None
        self.roi = []
        self.box = None
        self.img_stack_w_roi = []
        self.kymograph = []
        self.rescaling = False

    def generate_kymograph(self):
        self.kymograph = []

        if self.rescaling:
            for image in self.img_stack:
                img = []
                for coords in self.roi:
                    img.append(image[coords[0], coords[1]])
                img = rescale_intensity(img_as_float(img))
                cmap = plt.get_cmap("jet")
                rgba_img = cmap(img)
                self.kymograph.append(img)
        else:
            for image in self.img_stack:
                img = []
                for coords in self.roi:
                    img.append(image[coords[0], coords[1]])
                img = img_as_float(img)
                cmap = plt.get_cmap("jet")
                rgba_img = cmap(img)
                self.kymograph.append(img)

        filename = asksaveasfilename()
        os.mkdir(filename)
        os.mkdir(filename + "\\frames")

        frames = self.generate_gif()
        for i in range(len(frames)):
            imsave(filename + "\\frames\\" + str(i) + ".tif", frames[i])

        x0, y0, x1, y1 = self.box
        imsave(filename + "\\kymograph_x" + str(x0) + "-" +str(x1) + "_y" + str(y0) + "-" + str(y1) + ".png", self.kymograph)


    def generate_kymograph_3px_average(self):
        self.kymograph = []

        if self.rescaling:
            for image in self.img_stack:
                img = []
                for coords in self.roi:
                    px0 = image[coords[0] - 1, coords[1]]
                    px1 = image[coords[0], coords[1]]
                    px2 = image[coords[0] + 1, coords[1]]
                    img.append(np.mean([px0, px1, px2]))
                img = rescale_intensity(img_as_float(img))
                cmap = plt.get_cmap("jet")
                rgba_img = cmap(img)
                self.kymograph.append(img)
        else:
            for image in self.img_stack:
                img = []
                for coords in self.roi:
                    px0 = image[coords[0] - 1, coords[1]]
                    px1 = image[coords[0], coords[1]]
                    px2 = image[coords[0] + 1, coords[1]]
                    img.append(np.mean([px0, px1, px2]))
                img = img_as_float(img)
                cmap = plt.get_cmap("jet")
                rgba_img = cmap(img)
                self.kymograph.append(img)


        filename = asksaveasfilename(defaultextension=".png")
        imsave(filename, self.kymograph)

    def generate_gif(self):
        
        x0, y0, x1, y1 = self.box

        new_stack = []

        for i in range(len(self.img_stack)):
            crop_original = self.img_stack[i][x0-20:x1+20, y0-20:y1+20]
            crop_roi = self.img_stack_w_roi[i][x0-20:x1+20, y0-20:y1+20]
            new_stack.append(np.append(crop_roi, gray2rgb(crop_original), axis=1))

        return new_stack


if __name__ == "__main__":
    interface = Interface()
    interface.gui.mainloop()
