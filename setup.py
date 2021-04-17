"""setup module of two4two."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="two4two",
    version="0.0.4",
    author="Martin Schuessler, Leon Sixt, Philipp Weiss",
    author_email="dev@mschuessler.de",
    description="Generate biased image data to train and test classifiers.",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mschuessler/two4two",
    packages=setuptools.find_packages(),
    package_data={
        "": ["*.sh"],
        "two4two": ["py.typed"],
    },
    entry_points={
        'console_scripts': [
            'two4two_render_dataset=two4two.cli_tool:render_dataset',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    install_requires=[
        'imageio',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'toml',
        'tqdm',
    ],
    extras_require={
        'dev': [
            'flake8',
            'flake8-annotations',
            'flake8-docstrings',
            'flake8-import-order',
            'mypy',
            'pdoc',
            'pytest',
            'pytest-cov',
            'torch',
            'torchvision',
        ],
        'example_notebooks_data_generation': [
            'numpy',
            'pandas',
            'notebook'
        ],
        'example_notebooks_model_training': [
            'tensorflow',
            'pandas',
            'notebook',
            'livelossplot'
        ]
    },
    python_requires='>=3.7'
)
