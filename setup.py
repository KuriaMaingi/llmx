from setuptools import setup, find_packages

setup(
    name="llmx",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "llmx": ["configs/*.yml"],
    },
    install_requires=[
        # List your dependencies here
        "requests",
        # Add any other required packages
    ],
)