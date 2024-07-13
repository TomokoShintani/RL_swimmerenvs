from setuptools import setup

setup(
    name='gym_Smyswimmer3diffrgradv', #同層のディレクトリ名
    version='0.0.9',
    install_requires=[
        "numpy >= 1.18.0",
        "cloudpickle >= 1.2.0",
        "importlib_metadata >= 4.8.0; python_version < '3.10'",
        "gym_notices >= 0.0.4",
        "dataclasses == 0.8; python_version == '3.6'",
    ],
)