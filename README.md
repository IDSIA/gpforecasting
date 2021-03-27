# Timeseries forcasting with gaussian processes

## forgp package
The software include a small package that builds the gaussian process and uses it to produce some predictions. The package hevily depends on GPy. 
A convenience script can be used to run the GP over a large number of timeseries.

## **forecast&#46;py**
__forecast&#46;py__ is an executable python script that can be used to produce forecasts and evaluation of our GP over a number of different timeseries. The script takes as input a csv file containing training and test series and produces a csv file with predictions and scores. Input and output file are described below. 
The produced prediction includes mean and upper bound of the 95% confidence interval.

A number of command line arguments can be used to specify custom names for the columns in the input CSV file and to filter the timeseries to be processed. 
Most useful command line arguments are:

 * --frequency: to include timeseries with specific frequency
 * --normalize: normalize timeseries using the specified mean and standard deviation
 * --log: verbosity level (100 = max vebosity, 0 = min verbosity)
 * --default-priors to use default values for the priors in place place of no prior values
 * --help returns a description for the various paramters


## Input File format
Our tools use a simple tabular data format serialized in a CSV file for the training and test set.
This CSV files use as the only supported field separator is the comma ",". The first line contains the header, while Each following line in the file represents a timeseries. Required fields include:

 * __st__: unique name of the timeseries
 * __period__: frequency of the timeseries. One of MONTHLY, QUARTERLY, YEARLY and WEEKLY
 * __mean__: mean for the normalization of the timeseries
 * __std__: standard deviation for the normalization of the timeseries
 * __x__: training values of the timeseries
 * __xx__: test values of the timeseries

Point values of the timeseris are store as a semicolon (";") separated list of numeric values. 

Some of the column names may be specified as arguments to the script.

## Output file format
The output file follows a similar format as the input. It archived the predicted point forecast and 95% upperbounds a semicolon separated lists within a comma separated file where each line represent the corresponding timeseries in the input file.
The generated columns include:

 * __st__: the unique id of the series
 * __mean__: the mean of the training timeseries
 * __std__: the standard deviation of the training timeseries
 * __center__: the mean value of the prediction
 * __upper__: the upper bound of the 95% confidence prediction band 
 * __time__: time required to fit and predict
 * __mae__: mean absolute error of the predicted values (the xx values in the input file)
 * __crps__: continuous ranked probability score
 * __ll__: loglikelihood 

## Priors file
Prior values for the different kernel hyper parameters can be provided via file. This file contains the priors as a new line separated list of values. These are in order:
    
 * standard deviation of variances 
 * standard deviation of lengtscales
 * mean of variances
 * mean of rbf lenghtscale 
 * mean of periodic kernel's lenghtscale
 * mean of first spectral kernel's exponential component lenghtscale
 * mean of first spectral kernel's cosine component lenghtscale
 * mean of second spectral kernel's exponential component lenghtscale
 * mean of second spectral kernel's cosine component lenghtscale
 
The latter may be used only depending on the desired number of spectral components (Q parameter).

## Dependencies and setup
A requirements file is provided in the package to ease the installation of all the dependencies. For conda based systems one may create a suitable environment with:

```sh 
conda create --name <env> --file requirements.txt
```

## Example execution
The package includes a number of input files inluding standard M1, M3 competition timeseries, a sample of the M4 competition and a short example input.
To run the script on the example input one may run from withing the src folder the following command:

```sh
./forecast.py --log 100 --default-priors --normalize ../data/example_input example_output
```

As hinted above you may provide priors also via file:

```sh
./forecast.py --log 100 --default-priors --normalize --priors ../data/example_priors ../data/example_input example_output
```