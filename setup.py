from distutils.core import setup, Extension
import numpy

setup(
    name='nuclitrack',
    version='1.2.3',
    description='Nuclei tracking program',
    author='Sam Cooper',
    author_email='sam@socooper.com',
    license='MIT',
    packages=['nuclitrack'],
    ext_modules=[
        Extension("ctooltracking", ["nuclitrack/ctooltracking.c"], include_dirs=[numpy.get_include()]),
        Extension("classifyim", ["pyx_files/classifyim.c"], include_dirs=[numpy.get_include()]),
        Extension("ctoolsegmentation", ["nuclitrack/ctoolsegmentation.c"], include_dirs=[numpy.get_include()]),
        Extension("numpytoimage", ["nuclitrack/numpytoimage.c"], include_dirs=[numpy.get_include()])]
)
