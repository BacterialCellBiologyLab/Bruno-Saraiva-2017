"""Placeholder"""

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_isodata
from skimage.util import img_as_float
import tkFileDialog as FD
import numpy as np

class ImageManager(object):
    """Placeholder"""

    def __init__(self):

        self.border = 10
        self.clip = None

        self.phase_image = None
        self.phase_name = None

        self.channel_1_image = None
        self.channel_1_name = None

        self.channel_2_image = None
        self.channel_2_name = None

        self.cells_mask = None

        self.rescale_intensity = True

        self.working_dir = None

    def load_images(self):
        """Placeholder"""

        self.load_phase_image()
        self.create_cells_mask()
        self.load_channel_1_image()
        self.load_channel_2_image()

        if self.rescale_intensity:
            self.channel_1_image = rescale_intensity(self.channel_1_image)
            self.channel_2_image = rescale_intensity(self.channel_2_image)

    def load_phase_image(self):
        """Loads the phase image and converts it to a gray-scale image"""
        filename = FD.askopenfilename(initialdir=self.working_dir, title="Load the Phase Image")
        self.working_dir = filename.split("/")[:len(filename.split("/"))-1]
        self.phase_name = filename.split("/")[-1]

        tmp_img = imread(filename)
        tmp_img = rgb2gray(tmp_img)

        self.phase_image = tmp_img

        x_length, y_length = self.phase_image.shape
        self.clip = (self.border, self.border,
                     x_length - self.border, y_length - self.border)

    def load_channel_1_image(self):
        """Loads the channel 1 image and converts it to a gray-scale image"""

        filename = FD.askopenfilename(initialdir=self.working_dir, title="Load the Channel 1 Image")
        self.channel_1_name = filename.split("/")[-1]

        tmp_img = imread(filename)
        if len(tmp_img.shape) > 2:
            tmp_img = rgb2gray(tmp_img)

        tmp_img = img_as_float(tmp_img)

        mask = self.cells_mask

        best = (0, 0)
        x0, y0, x1, y1 = self.clip

        minscore = 0
        width = 10
        for dx in range(-width, width):
            for dy in range(-width, width):
                tot = -np.sum(np.multiply(mask,
                                          tmp_img[x0 + dx:x1 + dx,
                                                  y0 + dy:y1 + dy]))

                if tot < minscore:
                    minscore = tot
                    best = (dx, dy)

        dx, dy = best
        self.channel_1_image = tmp_img[x0 + dx:x1 + dx, y0 + dy:y1 + dy]

    def load_channel_2_image(self):
        """Loads the channel 2 image and converts it to a gray-scale image"""

        filename = FD.askopenfilename(initialdir=self.working_dir, title="Load the Channel 2 Image")
        self.channel_2_name = filename.split("/")[-1]

        tmp_img = imread(filename)
        if len(tmp_img.shape) > 2:
            tmp_img = rgb2gray(tmp_img)

        tmp_img = img_as_float(tmp_img)

        mask = self.cells_mask

        best = (0, 0)
        x0, y0, x1, y1 = self.clip

        minscore = 0
        width = 10
        for dx in range(-width, width):
            for dy in range(-width, width):
                tot = -np.sum(np.multiply(mask,
                                          tmp_img[x0 + dx:x1 + dx,
                                                  y0 + dy:y1 + dy]))

                if tot < minscore:
                    minscore = tot
                    best = (dx, dy)

        dx, dy = best

        self.channel_2_image = tmp_img[x0 + dx:x1 + dx, y0 + dy:y1 + dy]

    def create_cells_mask(self):
        """Placeholder"""

        x0, y0, x1, y1 = self.clip
        threshold_value = threshold_isodata(self.phase_image)
        self.cells_mask = (self.phase_image < threshold_value)[x0:x1, y0:y1]
