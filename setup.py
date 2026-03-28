"""Setup configuration for Movie Recommendation OpenEnv project"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="movie-recommendation-env",
    version="0.1.0",
    author="Your Name",
    description="An OpenEnv implementation for training movie recommendation agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MovieRecommendationEnv",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "trl[vllm]>=0.7.0",
        "peft>=0.7.0",
        "accelerate>=0.24.0",
        "huggingface-hub>=0.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "ipynb>=0.5.1",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0",
        ],
    },
)
