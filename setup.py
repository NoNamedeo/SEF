from setuptools import find_packages, setup

setup(
    name="sef",
    version="0.2.0",
    description="Signal extraction framework for video and image sequences",
    author="Alejandro Innocenzi, Matteo Vittori",
    packages=find_packages(include=("library*", "sef*"), exclude=("tests", "ui")),
    install_requires=[
        "matplotlib",
        "numpy",
        "opencv-contrib-python",
        "PyYAML>=6.0",
        "streamlit",
        "ultralytics",
    ],
    entry_points={
        "console_scripts": [
            "sef=sef.cli:main",
        ],
    },
)
