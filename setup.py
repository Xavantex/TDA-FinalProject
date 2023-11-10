import setuptools
from pathlib import Path

with open(Path("README.md"), "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="topology",
    version="0.1",
    author="xavante",
    author_email="xavante.erickson@gmail.com",
    description="topology course",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #entry_points = {
    #    'console_scripts': ['convert=convert.convert:main'],
    #},
    packages=['stablerank'],#setuptools.find_packages(where='src', exclude=["main"]),
    package_dir={'stablerank': 'stablerank'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8',
    install_requires=[
        'wheel==0.41.2',
        'Cython==3.0.2',
        'Ripser==0.6.4',
        'pandas==2.0.3',
        'ipywidgets==8.1.1',
        'matplotlib==3.7.3',
        'wget==3.2',
    ]
)
