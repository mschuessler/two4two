BLENDER_DOWNLOAD=https://ftp.halifax.rwth-aachen.de/blender/release/Blender2.83/blender-2.83.9-linux64.tar.xz
PIP_DOWNLOAD=https://bootstrap.pypa.io/get-pip.py

PYTHON_PATH=blender/2.83/python/bin/

PIP3=$(PYTHON_PATH:=pip3)
PYTHON3=$(PYTHON_PATH:=python3.7m)

all: blender $(wildcard two4two/**/*)
	cd two4two
	sed -i "s|^\(\ *self.package_directory\ *=\ *\)'.*'$|\1'`pwd`'|" blender.py
	cd ..
	python setup.py install
	$(PYTHON3) setup.py install

blender: blender.tar.xz get-pip.py
	tar -xf blender.tar.xz
	mv blender-2.83.9-linux64 blender
	$(PYTHON3) get-pip.py
	$(PIP3) install -U pip
	$(PIP3) install numpy scipy matplotlib ipykernel

blender.tar.xz:
	curl $(BLENDER_DOWNLOAD) -o blender.tar.xz

get-pip.py:
	curl $(PIP_DOWNLOAD) -o get-pip.py

clean:
	rm -f blender.tar.xz
	rm -f get-pip.py

remove: clean
	rm -rf blender dist build two4two_laserschwelle.egg-info
