"""
Package installation script for FiberCleaveProcessing.

Defines metadata, dependencies, and entry points for
building and distributing the Fiber Cleave Processing CLI/toolkit.
"""

from setuptools import find_packages, setup

setup(
    name="FiberCleaveProcessing",
    version="0.1.0",
    description="Fiber cleave quality classifier and tension predictor using CNN + MLP models",
    author="Chris Lombardi",
    author_email="clombardi23245@gmail.com",
    url="https://github.com/c-lombardi23/ImageProcessing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow==2.19.0",
        "keras-tuner==1.4.7",
        "numpy>=2.1.0",
        "pandas>=2.3.0",
        "scikit-learn>=1.7.0",
        "joblib>=1.5.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.10.0",
        "pydantic>=2.0.0",
        "opencv-python>=4.8.0",
        "typer>=0.16.0",
        "click>=8.2.0",
        "xgboost>=3.0.0",
        "mlflow>=3.1.1",
        "gymnasium==1.1.1",
        "stable_baselines3==2.6.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.29.0",
            "ipython>=8.37.0",
            "pytest-mock>=3.10.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    include_package_data=True,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "cleave-app=cleave_app.main:main",
        ],
    },
    keywords="machine-learning computer-vision fiber-optics image-processing cnn mlp",
    project_urls={
        "Bug Reports": "https://github.com/c-lombardi23/ImageProcessing/issues",
        "Source": "https://github.com/c-lombardi23/ImageProcessing",
    },
)
