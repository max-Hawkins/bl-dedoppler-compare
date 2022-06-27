import bifrost as bf
import sys

filenames = ["/datag/blpd0/datax/blmeerkat_uk_copy/20200813/m063_guppi_59074_56465_004550_J0835-4510_0001.rawspec.0001.fil"]
# filenames = ["/mnt_home/mhawkins/fdmt/pulse_linear.fil"]


print("Building pipeline")
data = bf.blocks.read_sigproc(filenames, gulp_nframe=128)
data = bf.blocks.copy(data, 'cuda', unpack=False)
data = bf.blocks.transpose(data, ['pol', 'freq', 'time'])
data = bf.blocks.fdmt(data, max_dm=100.)
data = bf.blocks.copy(data, 'cuda_host')
# bf.blocks.write_sigproc(data)

print("Running pipeline")
bf.get_default_pipeline().run()
print("All done")