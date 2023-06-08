from setuptools import setup

setup(
    name="newton_smoothers",
    version="0.1.0",
    description="Second-order iterated smoothers for state estimation",
    author="Hany Abdulsamad, Fatemeh Yaghoobi",
    author_email="hany@robot-learning.de",
    install_requires=["numpy", "scipy", "jax", "jaxlib", "matplotlib"],
    packages=["newton_smoothers"],
    zip_safe=False,
)
