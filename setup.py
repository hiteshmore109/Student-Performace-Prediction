from setuptools import find_packages, setup
from typing import List


def get_requirements(filepath: str) -> List[str]:
    """
    Reads a file containing a list of requirements and returns them as a list of strings.

    Args:
        filepath (str): The path to the file containing the requirements.

    Returns:
        List[str]: A list of strings representing the requirements.
    """
    with open(file=filepath) as file:
        requirements = file.readlines()
        requirements = [requirement.replace("\n", "") for requirement in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

        return requirements


setup(
    name="student-performance-prediction",
    version="0.0.1",
    author="Hitesh More",
    author_email="hitesh.omprakashmore@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
