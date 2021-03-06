import setuptools
import pathlib
import re
import ast


HERE = pathlib.Path(__file__).parent


def main():
    long_description = read_description()
    version = read_version()
    print("Version:", repr(version))

    setuptools.setup(
        name='pyndarray',
        version=version,
        author='Tamás László Hegedűs',
        author_email='tamas.laszlo.hegedus@gmail.com',
        description="A pure python library for manipulating multidimensional arrays, imitating some of numpy's interface.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/sorgloomer/pyndarray',
        packages=setuptools.find_packages(),
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Environment :: Console',
            'Topic :: System :: Filesystems',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        python_requires='>=3.6',

        license='MIT',
        install_requires=[
        ],
        keywords=['python', 'array', 'tensor'],
    )


def read_version():
    try:
        txt = (HERE / 'pyndarray' / '__init__.py').read_text('utf-8')
        version = re.findall(r"^__version__ = ([^\n]*)$", txt, re.M)[0]
        version = ast.literal_eval(version)
        return version
    except IndexError:
        raise RuntimeError('Unable to determine version.')


def read_description():
    with open("README.md", "r") as fh:
        return fh.read()


if __name__ == "__main__":
    main()
