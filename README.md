# two4two Image Dataset Generator

## Introduction
This program utilizes [blender](https://www.blender.org/) to generate image datasets.
Currently there are two classes.
They are smilar looking, abstract animals, made of eight blocks.
See [Section Classes](#classes) for details.
You can either install the module and create your own datasets or download an example dataset with 120,000 images.

## Installation
The programm can only run, if Blender 2.83 is installed in `two4two/blender`.
The programm was tested with version 2.83.9.
For a Linux system you can simply run `install_blender.sh`.
The script should work on macOS, but I never tested that.
The download link within the script is a mirror for Germany.
If any problems occur please see the script and reproduce the steps.
You need to put blender in the module folder and install pip for blender.
Then use pip to install numpy and scipy.
After installing blender, use `setuptools` to install the `two4two` module.
Run `python setup.py install`.

## Classes
The two classes for classifcation are *sticky* and *stretchy*. 

## Example Dataset