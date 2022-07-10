import setuptools


install_requires = ["numpy", "torch>=1.11", "functorch", "tqdm"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scod_regression",
    version="0.1.0",
    author="Apoorva Sharma, Christopher Agia",
    author_email="apoorva@stanford.edu, cagia@cs.stanford.edu",
    description="Equip PyTorch models with SCOD for OoD detection on regression tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agiachris/scod-module",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=install_requires,
)
