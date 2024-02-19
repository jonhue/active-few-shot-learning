from setuptools import setup

setup(
    name="afsl",
    version="0.1.0",
    description="Active Few-Shot Learning for Neural Networks",
    url="https://github.com/jonhue/afsl",
    author="Jonas Hübotter",
    author_email="jonas.huebotter@inf.ethz.ch",
    license="MIT",
    packages=["afsl"],
    install_requires=[
        "torch",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
