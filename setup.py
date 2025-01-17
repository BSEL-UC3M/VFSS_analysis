from setuptools import setup, find_namespace_packages

setup(name='VFSS',
      packages=find_namespace_packages(include=["VFSS"]),
      description='VFSS: Videofluoroscopic Swallowing Studies automatic analysis',
      url='https://github.com/BSEL-UC3M/DLUS',
      author='IGT - BSEL-UC3M',
      author_email='lcubero@ing.uc3m.es',
      install_requires=[
          "MedPy",
          "nnunet==1.7.1",
          "pandas",
          "scikit-image",
          "scipy",
          "SimpleITK",
          "torch",
      ],
      keywords=['deep learning', 'dysphagia', 'medical image segmentation', 'VFSS']
      )
