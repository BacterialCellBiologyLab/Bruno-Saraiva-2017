from coloc import CoLoc


cl = CoLoc()
cl.load_images()
cl.process_cells()
cl.pearsons_test_percentile()
cl.select_cells()
cl.generate_pearsons_report_percentile()
