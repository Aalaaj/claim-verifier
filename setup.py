"""Setup configuration for thesis-claim-analyzer package"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="thesis-claim-analyzer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="A framework for detecting and analyzing claims in research papers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/thesis-claim-analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "claim-analyzer=scripts.run_analysis:main",
            "claim-train=src.training.train_claim_detector:main",
        ],
    },
)