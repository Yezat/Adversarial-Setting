from setuptools import setup
from setuptools.extension import Extension
import Cython.Build

ext_modules = [
    Extension(
        "brentq",
        sources=["numerics/brentq.c"],  # or ["your_module.pyx"] if you use Cython
        include_dirs=["numerics"],  # Directory for your .h file
        extra_compile_args=["-O3"],
        language="c",
    )
]

setup(
    name="adversarial-setting",
    ext_modules=Cython.Build.cythonize(ext_modules),
    packages=[
        "erm",
        "sweep",
        "model",
        "numerics",
        "experiments",
        "state_evolution",
        "util",
        "tests",
    ],
)
