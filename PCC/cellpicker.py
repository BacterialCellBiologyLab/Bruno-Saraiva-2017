"""Placeholder"""
import numpy as np
import Tkinter as tk
import matplotlib.cm as cm
import tkMessageBox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from skimage.io import imread, imsave
from skimage.util import img_as_int, img_as_float
from skimage import exposure
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

class CellPicker(object):
    """Placeholder"""

    def __init__(self):
        self.selected_cells = [] 
        self.rejected_cells = []

        self.current_index = 0
        self.cell_ids = None
        self.cells = None

         # GUI
        self.main_window = tk.Tk()
        self.main_window.wm_title("Cellpicker")

        self.top_frame = tk.Frame(self.main_window)
        self.top_frame.pack(fill="x")

        self.middle_frame = tk.Frame(self.main_window)
        self.middle_frame.pack(fill="x")

        self.bottom_frame = tk.Frame(self.main_window)
        self.bottom_frame.pack(fill="x")

        self.previous_button = tk.Button(self.top_frame,
                                         text="Back",
                                         command=self.previous_cell)
        self.previous_button.pack(side="right")

        self.select_button = tk.Button(self.bottom_frame,
                                       text="Select Cell",
                                       command=lambda: self.select_cell(True))
        self.select_button.pack(side="left")

        self.reject_button = tk.Button(self.bottom_frame,
                                       text="Reject Cell",
                                       command=lambda: self.select_cell(False))
        self.reject_button.pack(side="left")

        # creates the figure canvas
        self.fig = plt.figure(figsize=(12, 8), frameon=True)
        self.canvas = FigureCanvasTkAgg(self.fig, self.middle_frame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side="top")

        self.ax = plt.subplot(111)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self.ax.axis("off")
        plt.autoscale(False)
        
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.middle_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill="both")
        
        self.ax.format_coord = self.show_nothing

    def key(self, event):
        if event.char == "1":
            self.select_cell(True)
        elif event.char == "d":
            self.select_cell(False)
        elif event.char == "b":
            self.previous_cell()

    def select_cells(self, cellmanager):
        
        self.cell_ids = sorted(cellmanager.cells.keys())
        self.cells = cellmanager.cells
        self.main_window.bind("<Key>", self.key)

        self.show_image()
        self.main_window.mainloop()

        cellmanager.cells = self.cells

    def show_nothing(self, x, y):
        
        return ""

    def show_image(self):

        self.ax.cla()
        
        current_image = self.cells[self.cell_ids[self.current_index]].cell_image

        self.ax.imshow(current_image, cmap=cm.gray)
        self.canvas.show()

    def select_cell(self, state):
        
        if state == True:
            self.cells[self.cell_ids[self.current_index]].selection_state = 1

        else:
            self.cells[self.cell_ids[self.current_index]].selection_state = 0

        if self.current_index < len(self.cell_ids)-1:
            self.current_index += 1
            self.show_image()

        else:
            if tkMessageBox.askokcancel("Quit", "Last cell, generate report?"):
                self.main_window.destroy()
            else:
                self.show_image()

    def previous_cell(self):

        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()
        else:
            pass