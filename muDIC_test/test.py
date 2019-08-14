import muDIC as dic
import os
cwd = os.getcwd()
path = cwd + r"/example_images"
image_stack = dic.image_stack_from_folder(path,file_type=".bmp")
mesher = dic.Mesher()
mesh = mesher.mesh(image_stack)
inputs = dic.DICInput(mesh,image_stack)
dic_job = dic.DICAnalysis(inputs)
results = dic_job.run()
fields = dic.Fields(results)
displ = fields.disp()
#true_strain = fields.true_strain()
print(displ)
viz = dic.Visualizer(fields,images=image_stack)
viz.show(field="Displacement", component = (1,1), frame = 1)
