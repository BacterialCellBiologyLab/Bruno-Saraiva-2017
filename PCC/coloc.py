"""Main module of the application, serves as the interface to interact
with every other module"""

from imagemanager import ImageManager
from cellmanager import CellManager
from colocprocessor import CoLocProcessor
from reportsmanager import ReportsManager
from cellpicker import CellPicker

class CoLoc(object):
    """Placeholder"""

    def __init__(self):
        self.image_manager = ImageManager()
        self.cell_manager = CellManager()
        self.coloc_processor = CoLocProcessor()
        self.reports_manager = ReportsManager()
        self.cell_picker = CellPicker()

    def load_images(self):
        """Placeholder"""

        self.image_manager.load_images()

    def create_mask(self):
        """Placeholder"""

        self.image_manager.create_cells_mask()

    def process_cells(self):
        """Placeholder"""

        self.cell_manager.process_cells(self.image_manager.cells_mask)

    def pearsons_test(self):
        """Placeholder"""

        self.coloc_processor.pearsons_test(self.image_manager, self.cell_manager)

    def pearsons_test_percentile(self):
        """Placeholder"""

        self.coloc_processor.cell_pearsons_test_percentile(self.image_manager,
                                                           self.cell_manager,
                                                           0.3)

    def select_cells(self):

        self.cell_picker.select_cells(self.cell_manager)

    def generate_pearsons_report(self):
        """Placeholder"""

        self.reports_manager.pearsons_report(self.image_manager, self.cell_manager,
                                             self.coloc_processor)

    def generate_pearsons_report_percentile(self):
        """Placeholder"""

        self.reports_manager.pearsons_report_percentile(self.image_manager,
                                                        self.cell_manager,
                                                        self.coloc_processor)
