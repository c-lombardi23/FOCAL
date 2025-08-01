Installation
============

Requirements
------------

* Python 3.10 or higher
* pip package manager


Install from Source
-------------------

First, clone the repository, create and activate a virtual environment, and then install the package in "editable" mode. This makes the `cleave-app` command available in your terminal.

.. code-block:: bash

   git clone http://github.com/c-lombardi23/ImageProcessingClone.git
   cd ImageProcessingClone

   # Create and activate a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # venv\Scripts\activate    # On Windows

   # Install the package and its development dependencies
   pip install -e ".[dev]"

Verify the Installation
-----------------------

After installation, run the following command to see all available options and confirm the CLI is working:

.. code-block:: bash
    
    cleave-app --help

Dependencies
------------

The package will automatically install these dependencies:

* numpy
* pandas
* scikit-learn
* tensorflow
* keras-tuner
* joblib
* Pillow
* pydantic
* opencv-python
* typer
* click
