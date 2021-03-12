import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CTRs", # Replace with your own username
    version="0.0.9",
    author="Tim Su",
    author_email="omg80827@gmail.com",
    description="A package for CTR prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimJLS/CTRs",
    project_urls={
        "Bug Tracker": "https://github.com/TimJLS/CTRs/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "tensorflow>=2"
    ]
)
