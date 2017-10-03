"""Placeholder"""
from scipy.stats import pearsonr
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_int
import numpy as np

class CoLocProcessor(object):
    """Placeholder"""

    def __init__(self):
        self.whole_image_pcc = None
        self.cells_pcc = None

    def pearsons_test(self, imagemanager, cellmanager):
        """Placeholder"""

        self.whole_image_pcc = self.whole_image_pearsons_test(imagemanager)
        self.cells_pcc = self.cell_pearsons_test(imagemanager, cellmanager)

    def pearsons_score(self, channel_1, channel_2, mask):
        """Placeholder"""

        filtered_1 = (channel_1 * mask).flatten()
        filtered_1 = filtered_1[filtered_1 > 0.0]
        filtered_2 = (channel_2 * mask).flatten()
        filtered_2 = filtered_2[filtered_2 > 0.0]

        return pearsonr(filtered_1, filtered_2)

    def cell_pearsons_test(self, imagemanager, cellmanager):
        """Placeholder"""

        pcc_scores = {}

        for key in cellmanager.cells.keys():
            cell = cellmanager.cells[key]
            x0, y0, x1, y1 = cell.box

            channel_1_cell = imagemanager.channel_1_image[x0:x1+1, y0:y1+1]
            channel_2_cell = imagemanager.channel_2_image[x0:x1+1, y0:y1+1]

            c1_overlay = mark_boundaries(img_as_int(channel_1_cell), img_as_int(cell.cell_mask),
                                         color=(1, 0, 1), outline_color=None)
            c2_overlay = mark_boundaries(img_as_int(channel_2_cell), img_as_int(cell.cell_mask),
                                         color=(1, 0, 1), outline_color=None)

            cell.cell_image = np.concatenate((c1_overlay,
                                              c2_overlay),
                                              axis=1)

            cell.cell_image_overlays = np.concatenate((c1_overlay, c2_overlay), axis=1)

            pcc_scores[key] = self.pearsons_score(channel_1_cell,
                                                  channel_2_cell,
                                                  cell.cell_mask)

        return pcc_scores

    def cell_pearsons_test_percentile(self, imagemanager, cellmanager, percentile):

        pcc_scores = {}

        for key in cellmanager.cells.keys():
            cell = cellmanager.cells[key]
            x0, y0, x1, y1 = cell.box

            channel_1_cell = imagemanager.channel_1_image[x0:x1+1, y0:y1+1]
            channel_2_cell = imagemanager.channel_2_image[x0:x1+1, y0:y1+1]

            c1_overlay = mark_boundaries(img_as_int(channel_1_cell), img_as_int(cell.cell_mask),
                                         color=(1, 0, 1), outline_color=None)
            c2_overlay = mark_boundaries(img_as_int(channel_2_cell), img_as_int(cell.cell_mask),
                                         color=(1, 1, 0), outline_color=None)

            cell.cell_image = np.concatenate((c1_overlay,
                                              c2_overlay),
                                              axis=1)

            channel_1_pxs = channel_1_cell * cell.cell_mask
            channel_1_pxs = channel_1_pxs.flatten()
            channel_1_pxs = sorted(channel_1_pxs)
            channel_1_pxs = np.trim_zeros(channel_1_pxs)[::-1]
            channel_1_pxs = channel_1_pxs [:int(len(channel_1_pxs)*percentile)]
            channel_1_threshold = channel_1_pxs[-1]

            channel_1_mask = channel_1_threshold <= (channel_1_cell * cell.cell_mask)

            channel_2_pxs = channel_2_cell * cell.cell_mask
            channel_2_pxs = channel_2_pxs.flatten()
            channel_2_pxs = sorted(channel_2_pxs)
            channel_2_pxs = np.trim_zeros(channel_2_pxs)[::-1]
            channel_2_pxs = channel_2_pxs [:int(len(channel_2_pxs)*percentile)]
            channel_2_threshold = channel_2_pxs[-1]

            channel_2_mask = channel_2_threshold <= (channel_2_cell * cell.cell_mask)

            c1_overlay_1 = mark_boundaries(img_as_int(c1_overlay), img_as_int(channel_1_mask),
                                         color=(1, 1, 0), outline_color=None)
            c2_overlay_1 = mark_boundaries(img_as_int(c2_overlay), img_as_int(channel_1_mask),
                                         color=(1, 1, 0), outline_color=None)

            c1_overlay_2 = mark_boundaries(img_as_int(c1_overlay), img_as_int(channel_2_mask),
                                         color=(1, 1, 0), outline_color=None)
            c2_overlay_2 = mark_boundaries(img_as_int(c2_overlay), img_as_int(channel_2_mask),
                                         color=(1, 1, 0), outline_color=None)

            cell.cell_image_overlays = np.concatenate((c1_overlay_1, c2_overlay_1,
                                              c1_overlay_2, c2_overlay_2), axis=1)

            #missing creation of mask for each channel and changes to allow the
            #computation of two pcc's

            percentile_mask = None

            pcc_scores[key] = (self.pearsons_score(channel_1_cell,
                                                   channel_2_cell,
                                                   channel_1_mask),
                               self.pearsons_score(channel_1_cell,
                                                   channel_2_cell,
                                                   channel_2_mask))

        self.cells_pcc = pcc_scores


    def whole_image_pearsons_test(self, imagemanager):
        """Placeholder"""

        return self.pearsons_score(imagemanager.channel_1_image,
                                   imagemanager.channel_2_image,
                                   imagemanager.cells_mask)
