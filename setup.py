from setuptools import setup, find_packages


setup(name='SDRL',
      version='1.0.0',
      description='Semantic-Aware Disentanglement Representation Learning for Unsupervised Person Re-identification',
      author='Xuefeng Tao',
      url='https://github.com/taoxuefong/SDRL-reid',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.3'],
      packages=find_packages()
      )