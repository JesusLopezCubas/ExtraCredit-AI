¡# setup.py
from setuptools import setup, find_packages

setup(
    name="extra_credit_ai",           # this is the pip install name
    version="0.1.0",
    author="Jesús López Cubas",
    author_email="jesuslopezcubas@gmail.com",
    description="Código ExtraCredit-AI convertido en un paquete instalable",
    packages=find_packages(),         # will find any folder containing __init__.py
    install_requires=[
        # List any runtime dependencies here, e.g. "numpy>=1.18.0"
    ],
    python_requires=">=3.6",
)
