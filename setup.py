'''
Author: ZJG
Date: 2022-06-17 09:34:18
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-10-14 10:31:41
'''
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import re
import sys

import setuptools


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TFCrypt"))

# Read description and requirements.
with open("README.md", encoding="utf8") as f:
    readme = f.read()
with open("requirements.txt") as f:
    reqs = f.read()

# get version string from module
init_path = os.path.join(os.path.dirname(__file__), "TFCrypt/__init__.py")
with open(init_path, "r") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

# Set key package information.
DISTNAME = "TFCrypt"
DESCRIPTION = "TFCrypt: secure machine learning in PyTorch."
LONG_DESCRIPTION = readme
AUTHOR = "ZJG"
LICENSE = "MIT licensed, as found in the LICENSE file"
REQUIREMENTS = (reqs.strip().split("\n"),)
VERSION = version

# Run installer.
if __name__ == "__main__":
    if sys.version_info < (3, 7):
        sys.exit("Sorry, Python >=3.7 is required for TFCrypt.")

    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(),
        dependency_links=[],
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        url="https://github.com/ZJG0/TFCrypt",
        author=AUTHOR,
        license=LICENSE,
        tests_require=["pytest"],
        data_files=[("/configs", ["configs/default.yaml"])],
    )
