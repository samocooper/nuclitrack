from distutils.core import setup, Extension
import numpy

setup(
    name='nuclitrack',
    version='1.1.4',
    description='Nuclei tracking program',
    author='Sam Cooper',
    author_email='sam@socooper.com',
    license='MIT',
    packages=['nuclitrack'],
    install_requires=['Cython','numpy','matplotlib','scipy','scikit-image','scikit-learn','kivy','h5py'],
    ext_modules=[
        Extension("tracking_c_tools", ["nuclitrack/ctooltracking.c"], include_dirs=[numpy.get_include()]),
        Extension("segmentation_c_tools", ["nuclitrack/ctoolsegmentation.c"], include_dirs=[numpy.get_include()]),
        Extension("numpy_to_image", ["nuclitrack/numpytoimage.c"], include_dirs=[numpy.get_include()])]
)
