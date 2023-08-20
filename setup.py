from setuptools import setup,find_packages
from typing import List

HYPHAN_E_DOT = "-e ."

def get_requirements(filepath:str)->[str]:
    requirement = []

    with open(filepath) as file_obj:
        requirement = file_obj.readlines()
        requirement = [i.replace("\n","" ) for i in requirement]

        if HYPHAN_E_DOT in requirement:
            requirement.remove(HYPHAN_E_DOT)
        
        return requirement



setup(
    name="Regressionprediction",
    version="0.0.1",
    author="yash mohite",
    author_email="mohite.yassh@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements("requirements.txt")
)
