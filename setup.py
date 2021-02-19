# coding=utf-8

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="two4two-laserschwelle",
    version="0.0.3",
    author="Philipp Weiss, Leon Sixt, Martin Schuessler",
    author_email="dev@mschuessler.de",
    description="Generate biased image data to train and test classifiers.",
    license='GPLv2+',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laserschwelle/two4two",
    packages=setuptools.find_packages(),
    package_data={
        "": "*.sh",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: POSIX :: Linux"
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            'flake8-import-order',
            'flake8-annotations',
            'flake8-docstrings',
        ]
    },
    python_requires='>=3.7'
)
