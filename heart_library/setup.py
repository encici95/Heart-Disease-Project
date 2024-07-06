from setuptools import setup, find_packages

setup(
    name='heart_library',                 # Name of your package
    version='0.1',                        # Version of your package
    packages=find_packages(),             # Automatically find and include all packages
    install_requires=[                    # List of dependencies
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'IPython',
        'sqlalchemy',
        'pymysql',
        'pyodbc'
    ],
    entry_points={                        # Define console scripts
        'console_scripts': [
            'heart_analyze=heart_library.cli:main',
        ],
    },
    author='ChiCuong EnCiCi Nguyen',                   # Author name
    author_email='chicuong.nguyen.encici@gmail.com', # Author email
    description='A library for heart disease analysis',  # Short description
    # url='https://github.com/', # URL to the project homepage
    classifiers=[                         # Additional metadata
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',              # Minimum Python version requirement
)