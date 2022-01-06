from setuptools import (
    find_packages,
    setup
)


with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()


setup(
    name="residual_connexion_analysis",
    version="0.1.0",
    author="Pierre-Arthur CLAUDE",
    packages=find_packages(exclude=["*tests*"]),
    entry_points={
        "console_scripts": [
            "resco_analysis = resco.__main__:main"
        ]
    },
    install_requires=[
        "tensorboard==2.7.0",
        "torch==1.10.1",
        "torchvision==0.11.2",
        "tqdm==4.62.3"
    ],
    tests_require=["pytest"]
)
