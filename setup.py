from setuptools import setup, find_packages


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(name='RL 2048',
      version='0.1.0_dev',
      description='Reinforcement learning to plat 2048',
      author='Vighnesh Birodkar',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='vighneshbirodkar@nyu.edu',
      )
