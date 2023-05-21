from setuptools import setup, find_packages

setup(
    name="hello-ml",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "pytest",
    ]
)
