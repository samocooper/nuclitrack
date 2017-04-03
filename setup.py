from setuptools import setup
from setuptools.extension import Extension
import numpy

setup(
    name='nuclitrack',
    version='1.0.8',
    description='Nuclei tracking program',
    author='Sam Cooper',
    author_email='sam@socooper.com',
    license='MIT',
    packages=['nuclitrack'],
    install_requires=['Cython','numpy','matplotlib','scipy','scikit-image','scikit-learn','pygame','kivy','h5py'],
    ext_modules=[
        Extension("tracking_c_tools", ["nuclitrack/tracking_c_tools.c"], include_dirs=[numpy.get_include()]),
        Extension("numpy_to_image", ["nuclitrack/numpy_to_image.c"], include_dirs=[numpy.get_include()])]
)
