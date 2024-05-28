from setuptools import setup, find_packages

setup(
    name='wdbo_algo',
    version='1.0.0',
    packages=find_packages(),
    author="Anthony Bardou",
    author_email="anthony.bardou@epfl.ch",
    description="W-DBO Algotithm for Dynamic Bayesian Optimization",
    install_requires=[
        "wdbo_criterion",
        "gpytorch",
        "torch",
        "botorch"
    ],
    python_requires='>=3.7',
    url='https://github.com/WDBO-ALGORITHM/wdbo_algo'
    project_urls={
        'W_DBO_criterion': 'https://github.com/WDBO-ALGORITHM/wdbo_criterion'
    }
)