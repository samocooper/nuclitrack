from distutils.core import setup, Extension
import numpy

setup(
    name='nuclitrack',
    version='2.03',
    description='Nuclei tracking program',
    author='Sam Cooper',
    author_email='sam@socooper.com',
    license='MIT',
    packages=['nuclitrack', 'nuclitrack.nuclitrack_gui', 'nuclitrack.nuclitrack_guitools',
              'nuclitrack.nuclitrack_tools', 'nuclitrack.nuclitrack_ctools'],
    ext_modules=[
        Extension("ctooltracking", ["nuclitrack/nuclitrack_ctools/ctooltracking.c"], include_dirs=[numpy.get_include()]),
        Extension("classifyim", ["nuclitrack/nuclitrack_ctools/classifyim.c"], include_dirs=[numpy.get_include()]),
        Extension("ctoolsegmentation", ["nuclitrack/nuclitrack_ctools/ctoolsegmentation.c"], include_dirs=[numpy.get_include()]),
        Extension("numpytoimage", ["nuclitrack/nuclitrack_ctools/numpytoimage.c"], include_dirs=[numpy.get_include()])]
)
