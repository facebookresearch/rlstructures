from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = [line.strip() for line in f]

setup(
    name="rlstructures",
    version="1.0",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=reqs,
)
