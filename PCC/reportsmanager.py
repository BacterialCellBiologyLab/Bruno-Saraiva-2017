"""Placeholder"""
from skimage.io import imsave
import tkFileDialog as FD
import numpy as np
import os

class ReportsManager(object):
    """Placeholder"""

    def __init__(self):
        pass

    def pearsons_report(self, imagemanager, cellmanager, colocprocessor):
        """Placeholder"""

        path = FD.askdirectory()
        report_path = path + "/" + imagemanager.phase_name + "_wholecell"
        images_path = report_path + "/_images"

        if not os.path.exists(report_path):
            os.mkdir(report_path)
            os.mkdir(images_path)

        cells = cellmanager.cells

        HTML_HEADER = """<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN"
                        "http://www.w3.org/TR/html4/strict.dtd">
                    <html lang="en">
                      <head>
                        <meta http-equiv="content-type" content="text/html; charset=utf-8">
                        <title>title</title>
                        <link rel="stylesheet" type="text/css" href="style.css">
                        <script type="text/javascript" src="script.js"></script>
                      </head>
                      <body>\n"""

        report = [HTML_HEADER]

        selected = "<h2>Selected Cells</h2>"
        table = '<table>\n<th>Cell ID</th><th>Images</th><th>PCC</th>'

        sorted_keys = []
        for k in sorted(cells.keys()):
            sorted_keys.append(int(k))

        sorted_keys = sorted(sorted_keys)

        for key in sorted(cellmanager.cells.keys()):
            cell = cellmanager.cells[key]

            if cell.selection_state == 1:

                cell_image = cell.cell_image_overlays

                imsave(images_path + "/" + key + ".png", cell_image)

                row = '<tr style="text-align:center"><td>' + key + '</td><td><img src="./' + '_images/' + key + '.png" alt="pic" width="200"/></td>' + \
                    "<td>" + str(colocprocessor.cells_pcc[key][0]) + "</td></tr>"

                table += row

        report += selected
        report += table
        report += "</table></body></html>"

        rejected = "<h2>Rejected Cells</h2>"
        table = '<table>\n<th>Cell ID</th><th>Images</th><th>PCC</th>'

        for key in sorted(cellmanager.cells.keys()):
            cell = cellmanager.cells[key]

            if cell.selection_state == 0:

                cell_image = cell.cell_image_overlays

                imsave(images_path + "/" + key + ".png", cell_image)

                row = '<tr style="text-align:center"><td>' + key + '</td><td><img src="./' + '_images/' + key + '.png" alt="pic" width="200"/></td>' + \
                    "<td>" + str(colocprocessor.cells_pcc[key][0]) + "</td></tr>"

                table += row

        report += rejected
        report += table
        report += "</table></body></html>"

        open(report_path + '/pcc_report.html', 'w').writelines(report)

    def pearsons_report_percentile(self, imagemanager, cellmanager, colocprocessor):
        """Placeholder"""

        path = FD.askdirectory()
        report_path = path + "/" + imagemanager.phase_name + "_percentile"
        images_path = report_path + "/_images"

        if not os.path.exists(report_path):
            os.mkdir(report_path)
            os.mkdir(images_path)

        cells = cellmanager.cells

        HTML_HEADER = """<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN"
                        "http://www.w3.org/TR/html4/strict.dtd">
                    <html lang="en">
                      <head>
                        <meta http-equiv="content-type" content="text/html; charset=utf-8">
                        <title>title</title>
                        <link rel="stylesheet" type="text/css" href="style.css">
                        <script type="text/javascript" src="script.js"></script>
                      </head>
                      <body>\n"""

        report = [HTML_HEADER]

        selected = "<h2>Selected Cells:</h2>"
        table = '<table>\n<th>Cell ID</th><th>C1 Region (c1 - c2) C2 Region (c1 - c2)</th><th>PCC Channel 1</th><th>PCC Channel 2</th>'

        sorted_keys = []
        for k in sorted(cells.keys()):
            sorted_keys.append(int(k))

        sorted_keys = sorted(sorted_keys)

        for key in sorted(cellmanager.cells.keys()):
            cell = cellmanager.cells[key]
            if cell.selection_state == 1:

                cell_image = cell.cell_image_overlays

                imsave(images_path + "/" + key + ".png", cell_image)

                row = '<tr style="text-align:center"><td>' + key + '</td><td><img src="./' + '_images/' + key + '.png" alt="pic" width="200"/></td>' + \
                    "<td>" + str(colocprocessor.cells_pcc[key][0][0]) + "</td>" + \
                    "<td>" + str(colocprocessor.cells_pcc[key][1][0]) + "</td></tr>"

                table += row

        report += selected
        report += table
        report += "</table></body></html>"

        rejected = "<h2>Rejected Cells:</h2>"
        table = '<table>\n<th>Cell ID</th><th>C1 Region (c1 - c2) C2 Region (c1 - c2)</th><th>PCC Channel 1</th><th>PCC Channel 2</th>'

        for key in sorted(cellmanager.cells.keys()):
            cell = cellmanager.cells[key]
            if cell.selection_state == 0:

                cell_image = cell.cell_image_overlays

                imsave(images_path + "/" + key + ".png", cell_image)

                row = '<tr style="text-align:center"><td>' + key + '</td><td><img src="./' + '_images/' + key + '.png" alt="pic" width="200"/></td>' + \
                    "<td>" + str(colocprocessor.cells_pcc[key][0][0]) + "</td>" + \
                    "<td>" + str(colocprocessor.cells_pcc[key][1][0]) + "</td></tr>"

                table += row

        report += rejected
        report += table
        report += "</table></body></html>"

        open(report_path + '/pcc_report.html', 'w').writelines(report)