"""
Setup script for the Simple Data Preparation Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simple-data-preparation-pipeline",
    version="0.1.0",
    author="KVS",
    description="A comprehensive data preparation pipeline for cleaning, transforming, and preparing data for machine learning and analysis tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simple-data-preparation-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "matplotlib>=3.10.3",
        "numpy>=2.2.6",
        "pandas>=2.3.1",
        "plotly>=6.2.0",
        "scikit-learn>=1.7.0",
        "seaborn>=0.13.2",
    ],
    extras_require={
        "dev": [
            "pytest>=8.4.1",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "data-pipeline=main:main",
        ],
    },
)