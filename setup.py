import setuptools

setuptools.setup(
    name="ssltsc",
    version="0.0.1",
    author="Jann Goschenhofer",
    author_email="jann.goschenhofer@stat.uni-muenchen.de",
    description="package for ssl on tsc",
    url="none.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    zip_safe=False
)
