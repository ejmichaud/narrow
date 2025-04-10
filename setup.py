from setuptools import setup, find_packages

setup(
    name="narrow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.30.0",
    ],
    author="Eric Michaud",
    author_email="eric.michaud99@gmail.com",
    description="Narrow: Code for training and evaluating narrow models",
    keywords="llama, transformers, nlp, machine learning",
    python_requires=">=3.7",
)