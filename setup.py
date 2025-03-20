from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path):
    "This function will return the list of required libraries"
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='End-To-End Data Science Project',
    author='Kaushik Iyer',
    author_email='iyerkaushik82@gmail.com',
    install_requires=get_requirements('requirements.txt')
)