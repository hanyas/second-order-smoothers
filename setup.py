from setuptools import setup

setup(name='second-order-smoothers',
      version='0.1.0',
      description='Second-order iterated smoothers for state estimation',
      author='Hany Abdulsamad, Fatemeh Yaghoobi',
      author_email='hany@robot-learning.de',
      install_requires=['jax', 'jaxlib', 'matplotlib'],
      packages=['second-order-smoothers'],
      zip_safe=False,
      )
