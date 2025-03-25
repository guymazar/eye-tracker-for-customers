from setuptools import setup, find_packages

setup(
    name='EyeTrackingProject',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'opencv-contrib-python',
    ],
) 