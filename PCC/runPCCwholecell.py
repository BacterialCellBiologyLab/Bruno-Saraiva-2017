from coloc import CoLoc


cl = CoLoc()
cl.load_images()
cl.process_cells()
cl.pearsons_test()
cl.select_cells()
cl.generate_pearsons_report()
