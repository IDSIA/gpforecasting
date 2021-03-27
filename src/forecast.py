#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
import numpy as np
import forgp.gp as gp 
import time 
import logging

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_col(data, name, index):
    return data.iloc[:,index] if index is not None else data.loc[:,name]

def parse_arguments():
    ''' Parse and validate command line arguments 
    '''
    parser = argparse.ArgumentParser(description="Time series forecasting using GP")
    parser.add_argument("--log", type=int, default=logging.WARNING)
    parser.add_argument('-Q', type=int, choices=[0, 1, 2], help="Number of spectral components", default=2)
    parser.add_argument('-r','--restarts', type=int, help="Number of restarts", default=1)
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--priors', help="Custom priors file location")
    group.add_argument('--default-priors', help="Use default priors", action="store_true")
    parser.add_argument('--priors-count', help="Number of blocks of priors in custom priors file location", default=1)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-xn', '--train-col', help="Name of the training data column (default x)", default="x")
    group.add_argument('-xi', '--train-index', help="Index (zero based) of the training data column")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-tn', '--test-col', help="Name of the test data column (default xx)", default="xx")
    group.add_argument('-ti', '--test-index', help="Index (zero based) of the test data column")

    parser.add_argument('--frequency-col', help="Specify the frequency column", default = "period")
    parser.add_argument('-f', '--frequency', help="Specify the frequency (monthly vs quarterly vs yearly)", default = "ANY")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mean-col', help="Specify the mean column name", default = "mean")
    group.add_argument('--mean-index', help="Specify the mean column index")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--std-col', help="Specify the std column name", default = "std")
    group.add_argument('--std-index', help="Specify the std column index")

    parser.add_argument('--normalize', help="should data be normalized manually", action='store_true')

    parser.add_argument('--limit', type=int, help="Specify the limit in the length of the training set (-1)", default=-1)

    parser.add_argument('--sample', help='limit to sample', type=int, default=-1)
    parser.add_argument('--sample-col', help='sample column', type=str, default="sample")

    parser.add_argument('data', help="Training test file (in csv format with arrays in semicolon separated)")
    parser.add_argument('target', help="Output file name")
    
    
    return parser.parse_args()


def float_arrays(data): 
    return data.str.split(";").apply(lambda x: np.array(x).astype(np.float))


def compute_indicators(Ytest, mean, upper):
    import properscoring as ps
    import scipy.stats as stat
    import numpy as np
    
    sigma = (upper - mean)/ stat.norm.ppf(0.975)
    fcast = mean
    
    crps = np.zeros(len(Ytest))
    ll = np.zeros(len(Ytest))
        
    for jj in range(len(Ytest)):
        crps[jj]    = ps.crps_gaussian(Ytest[jj], mu=fcast[jj], sig=sigma[jj])
        ll[jj]      = stat.norm.logpdf(x=Ytest[jj], loc=fcast[jj], scale=sigma[jj])
    
    mae = np.mean(np.abs(Ytest  - fcast))
    crps = np.mean(crps)
    ll = np.mean(ll)

    return([mae, crps, ll])


if __name__ == "__main__":
    args = parse_arguments()

    ## initialize logging
    logger = logging.getLogger()
    logger.setLevel(100 - args.log)
    ch = logging.StreamHandler()
    ch.setLevel(100- args.log)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Summary: GP forecasting with RBF, no ETS, no Bias")
    

    data = pd.read_csv(args.data)
    if args.frequency.upper() != "ANY":
        data = data[data[args.frequency_col].str.upper() == args.frequency.upper()]
        logger.info(f"Using only {args.frequency} data")
    else:
        logger.info("Not filterning by period")

    # filtering by sample number
    if args.sample > 0:
        data = data[data[args.sample_col] <= args.sample]
        logger.info(f"filtering {args.sample}: {data.shape}")

   
    train = get_col(data, args.train_col, args.train_index)
    test = get_col(data, args.test_col, args.test_index)
    
    train = float_arrays(train)
    test = float_arrays(test)

    if args.normalize: 
        means = get_col(data, args.mean_col, args.mean_index)
        stds = get_col(data, args.std_col, args.std_index)

        train = (train - means) / stds
        test = (test - means) / stds

    priors = None
    if args.priors is not None:
        priors = pd.read_csv(args.priors, header=None, comment='#')
        cut = int(len(priors) / args.priors_count)
        priors = priors[-cut:].values.flatten()
        pstr = str.join(" ", priors.astype(str))
        logger.debug(f"Using these custom priors: {pstr}")
    elif args.default_priors: 
        logger.info("Using default priors")
        priors = True
    else:
        logger.info("Using no priors")
        priors = False
    
    #priors = False

    

    out = pd.DataFrame(columns=["st", "mean", "std", "center", "upper"])
    for i in range(0, len(train)):
        start = time.time()

            
        row = data.iloc[i,:]
        Y = train.iloc[i]
        if args.limit > 0: 
            Y = Y[-args.limit:]
            logger.info(f"train length: {len(Y)}")
        
        Y = Y.reshape(len(Y), 1)
        YY = test.iloc[i]
        
        # Stderr output to be able to identify GPy errors 
        print(f"----------------------------------------------", file=sys.stderr)
        print(f"Processing series #{i}", file=sys.stderr)
        print(f"st: {row.st}", file=sys.stderr)
                
        logger.info(f"Processing series #{i}")
        logger.info(f"st: {row.st}")
        logger.debug(f"{row[args.frequency_col]}")
        logger.debug(f"series mean: {row[args.mean_col]}")
        logger.debug(f"series std: {row[args.std_col]}")
        logger.debug(f"train length: {len(Y)}/{len(train.iloc[i])}")
        logger.debug(f"test length: {len(YY)}")
        
        g = gp.GP(row[args.frequency_col].lower(), priors=priors, Q=args.Q, normalize=False, restarts=args.restarts)
        if type(priors) == "list":
            g.priors_array(priors)

        g.build_gp(Y)

        m, u = g.forecast(len(YY))
        m = m.reshape(len(m))
        u = u.reshape(len(u))

        mae, crps, ll = compute_indicators(YY, m, u)
        end = time.time()
        logger.debug(f"duration: {end - start}")

        out = out.append([{
            "st":row.st, 
            "mean":row[args.mean_col], 
            "std":row[args.std_col], 
            "time": end - start,
            "center":str.join(";", m.astype(str)), 
            "upper":str.join(";", u.astype(str)), 
            "mae": mae,
            "crps": crps,
            "ll": ll
        }])
        
out.to_csv(args.target)
