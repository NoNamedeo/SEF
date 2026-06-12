from setuptools import find_packages, setup

setup(
    name="sef",
    version="0.2.0",
    description="Signal extraction framework for video and image sequences",
    author="Matteo Vittori, Alejandro Innocenzi",
    packages=find_packages(include=("sef*",), exclude=("tests", "ui")),
    install_requires=[
        "matplotlib",
        "numpy",
        "opencv-contrib-python",
        "ultralytics",
        "PyYAML>=6.0",
        "streamlit",
    ],
    entry_points={
        "console_scripts": [
            "sef=sef.cli:main",
        ],
    },
)
