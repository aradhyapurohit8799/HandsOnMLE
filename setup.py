import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Housing_Price_Prediction",
    version="0.3",
    description="Assignment 4.1 Package for Housing Price Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aradhya-Purohit/mle-training.git",
    author="Aradhya Purohit",
    author_email="aradhya.purohit@tigeranalytics.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
