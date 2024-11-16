from setuptools import find_packages, setup

setup(
    name="pipecat-flows",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "aiohttp",
    ],
)
