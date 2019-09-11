import muDIC as dic
import numpy as np
import muDIC.vlab as vlab
    
image_shape = (2000, 2000)
speckle_image = vlab.rosta_speckle(image_shape, dot_size=4, density=0.32, smoothness=2.0)

F = np.array([[1.1, .0], [0., 1.0]], dtype=np.float64)
image_deformer = vlab.imageDeformer_from_defGrad(F)
downsampler = vlab.Downsampler(image_shape=image_shape, factor=4, fill=0.8, pixel_offset_stddev=0.1)
noise_injector = vlab.noise_injector("gaussian", sigma=.1)
image_generator = vlab.VirtualExperiment(speckle_image=speckle_image, image_deformer=image_deformer,
                          downsampler=downsampler, noise_injector=noise_injector, n=n)
image_stack = dic.ImageStack(image_generator)

mesher = dic.Mesher(deg_n=1,deg_e=1)
mesh = mesher.mesh(image_stack)

input = muDIC.solver.correlate.DIC_input(mesh, image_stack)
dic_job = dic.DIC_analysis(input)
results = dic_job.run()
