# CITS3200_Group16
Digital Image Correlation in Python 3. Using spline interpolation and Newton-Raphson method for convergence.

## Setup
To setup & install dependencies we will create a virtual environment and install from `requirements.txt`.

First run
`python3 -m venv venv` 
to create a virtual environment, then
`python3 -m pip install -r requirements.txt`
to install the necessary packages into the virtual environment.

## Using in a program
From the `deformation_measurement` package, import the class DIC_NR.

In code you create it, then supply it with the parameters in `set_parameters` to calculate deformation from.

These parameters are the `reference image`, `deformed image`, `subset size`, and `initial guess`.

After that, the method `calculate` will return the results as a numpy array.

For example:
```python3
import deformation_measurement as dm

dic = dm.DIC_NR()
dic.set_parameters("ref_image.bmp", "def_image.bmp", 11, [0, 0])
results = dic.calculate()

print(results)
```

## Using from the command line
A helpful script is included in the root directory of this repo named `measure_deformation.py`.

To run it with default settings, mark it as executable and then use `./measure_deformation.py ref_image.bmp def_image.bmp`.

For an explanation of all the parameters run `./measure_deformation.py -h`.

## Testing
Run `python test` to run the full test suite.

For testing a specific file you can use `python test Test_C_First_Order` or `python test Test_DIC_NR`.
