import setuptools

setuptools.setup(
    name='datarobot',
    version='2.25.1',
    packages=setuptools.find_packages(),
    install_requires=[
        "contextlib2>=0.5.5"
        "pandas<1.3.1,>=0.15"
        "pyyaml>=3.11"
        "requests>=2.21"
        "requests-toolbelt>=0.6"
        "trafaret!=1.1.0,<2.0,>=0.7"
        "urllib3>=1.23"
        "attrs<21.0,>=19.1.0"
    ],
    extras_require={
        'dev': [
            "mock==3.0.5"
            "pytest<5,>=4.6"
            "pytest-cov==3.0.0"
            "responses<0.10,>=0.9"
            "flake8<3,>=2.5.2"
            "Sphinx==1.8.3"
            "sphinx-rtd-theme==0.1.9"
            "nbsphinx<1,>=0.2.9"
            "nbconvert==5.3.1"
            "numpydoc>=0.6.0"
            "tox==3.24.4"
            "jupyter-contrib-nbextensions"
            "tornado<6.0"
            "decorator<5;python_version=='2.7'"
            "black==19.10b0;python_version>='3.6'"
            "isort==5.8;python_version>='3.6'"
        ],
        'examples': [
            "jupyter<=5.0"
            "fredapi==0.4.0"
            "matplotlib>=2.1.0"
            "seaborn<=0.8"
            "sklearn<=0.18.2"
            "wordcloud<=1.3.1"
            "colour<=0.1.4"
            "decorator<5;python_version=='2.7'"
        ],
        'lint': [
            "black==19.10b0;python_version>='3.6'"
            "isort==5.8;python_version>='3.6'"
        ]
    }
)
