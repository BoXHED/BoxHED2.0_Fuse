import os
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from boxhed.utils import timer
import matplotlib.pyplot as plt
from boxhed.boxhed import boxhed
from boxhed.model_selection import cv, best_param_1se_rule


def dump_pickle(obj, addr):
    with open(addr, 'wb') as handle:
        pickle.dump(obj, handle) 


def load_pickle(addr):
    with open(addr, 'rb') as handle:
        obj = pickle.load(handle)
    return obj 


# a helper function for sectioning the output in the terminal
def print_in_terminal (msg):
    def print_fancy_lines():
        for _ in range(3):
            print ("#" + " " + 116*"=" + " " + "#")

    print_fancy_lines()
    print (int(0.5*(120-len(msg)))*" "+msg)
    print_fancy_lines()


# This function creates the 3D plot of hazard
def section_4_plot_hazard_estimations(boxhed_, test_X, figaddr):

    # get hazard values on the test set
    haz_vals = boxhed_.hazard(test_X)

    figsize  = (6,6)
    dpi      = 150
    fontsize = 13

    fig  = plt.figure(figsize=figsize, dpi=dpi)
    ax   = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(*[arr.reshape(*([int(math.sqrt(len(haz_vals)))]*2)) for arr in [test_X['t'].values, test_X['X_0'].values, haz_vals]],
                        linewidth=0, antialiased=False)

    plt.title("Estimated Hazard Values", fontsize=fontsize)
    plt.ylabel(r"$X_0$", fontsize=fontsize, rotation=0, labelpad=30)
    plt.xlabel("time",   fontsize=fontsize, rotation=0, labelpad=30)

    plt.xticks(fontsize= fontsize-7)
    plt.yticks(fontsize= fontsize-7)

    plt.show()
    plt.savefig(figaddr)
    print (">>>", f"Hazard was plotted and saved as {figaddr}.", end="\n\n")


# This function creates the estimated survivor curve
def section_4_plot_survivor(boxhed_, df_surv, figaddr):
    surv_vals = boxhed_.survivor(df_surv) # Estimate the value the survivor curve for each row of df_surv

    figsize  = (3.5,3.5)
    dpi      = 150
    fontsize = 13

    fig  = plt.figure(figsize=figsize, dpi=dpi)
    ax   = fig.add_subplot()
    ax.plot (df_surv['t'].values, surv_vals)

    plt.title("Estimated survivor curve", fontsize=fontsize)
    plt.ylabel(r"$\hat S(t|x_0,x_1,\cdots)$", fontsize=fontsize, rotation=90, labelpad=30)
    plt.xlabel("time",   fontsize=fontsize, rotation=0, labelpad=30)

    plt.xticks(fontsize= fontsize-7)
    plt.yticks(fontsize= fontsize-7)

    plt.xlim([0, 1])
    plt.ylim([0.3, 1])

    plt.show()
    plt.savefig(figaddr)
    print (">>>", f"Survivor curve was plotted and saved as {figaddr}.", end="\n\n")


# This function downloads training data for our scalability analysis.
def section_5_download_train_data(train_data_addr):
    import gdown
    gdown.download(id="11Mh9JrXBnmEmNX0C1JvaTUKDuksq9KX7", output = train_data_addr, quiet=False)


# This function times boxhed and saves the runtimes in a text file
def section_5_boxhed_runtime(train_data_addr, runtimes_file):
    nthread = 20

    nrows = [2e6, 4e6, 6e6, 8e6, 10e6]
    nrow_to_sub = {
        int(2e6):  93193,
        int(4e6):  186333,
        int(6e6):  279512,
        int(8e6):  372658,
        int(10e6): 465804,
    }

    # read the training data
    train_data = pd.read_csv(train_data_addr, compression="gzip")

    runtime_dict = {}
    for nrow in tqdm(nrows, desc="Measuring BoXHED runtimes"):
        # create a boxhed instance
        boxhed_ = boxhed(nthread=nthread, max_depth = 1, n_estimators = 250)

        # start runtime
        timer_  = timer()

        # preprocess the data
        X_post = boxhed_.preprocess(data = train_data.head(nrow_to_sub[int(nrow)]), # subset the data based on the number of rows
                                num_quantiles = 256, 
                                weighted      = False, 
                                nthread       = nthread)

        # fit the boxhed instance to the preprocessed data
        boxhed_.fit(X_post['X'], X_post['delta'], X_post['w'])

        duration                = timer_.get_dur()
        runtime_dict [int(nrow)] = duration

        del boxhed_
    
    # writing results to a text file
    with open(runtimes_file, 'w') as f:
        for nrow in map(int, nrows):
            f.write(f"{nrow}:{runtime_dict[nrow]}\n")


# This function calls the R script which fits a blackboost instance to the data and writes the runtimes to a text file
def section_5_blackboost_runtime(blackboost_runtimes_script, train_data_addr, runtimes_file):
    import subprocess
    p = subprocess.Popen(f"Rscript {blackboost_runtimes_script} --trainDataAddr={train_data_addr} --runtimesFile={runtimes_file}", shell=True)
    p.wait()


# This function plots the runtimes of boxhed and blackboost
def section_5_plot_boxhed_blackboost_runtimes (boxhed_runtimes_file, blackboost_runtimes_file, figaddr):

    # This function reads in the runtime files and creates two numpy arrays of number of rows and the corresponding values
    def runtimes_as_dict(fname):
        with open(fname) as f:
            lines = [line.rstrip().split(sep=":") for line in f]
        nrows, runtimes = map(list, zip(*lines))
        return [np.array(list(map(type, arr))) for arr, type in [[nrows, int],[runtimes, float]]]

    boxhed_nrows,     boxhed_runtimes     = runtimes_as_dict(boxhed_runtimes_file)
    blackboost_nrows, blackboost_runtimes = runtimes_as_dict(blackboost_runtimes_file)
    assert np.array_equal(boxhed_nrows, blackboost_nrows) #assert that the runtimes are for the same number of rows
    nrows                                = boxhed_nrows

    plot_configs = [
        {
            'label':  "BoXHED2.0",
            'runtimes': boxhed_runtimes,
            'color':   "k" ,
            'marker':  "s",
            'lw':       2,
            'ls':      (0, (5, 10)),
            's':        600
        },
        {
            'label':  "Blackboost",
            'runtimes': blackboost_runtimes,
            'color':   "blue",
            'marker':  "o",
            'lw':       2,
            'ls':      "solid",
            's':        600
        }
    ]

    font_size = 40
    plt.figure(figsize=(19,12), dpi=700)
    plt.subplots_adjust(left=0.14, bottom=0.14, right=0.98, top=0.98)
    plt.xticks(fontsize= font_size)
    plt.yticks(fontsize= font_size)
    plt.xlabel("# rows (in millions)", fontsize=font_size)
    plt.ylabel("time (sec)", fontsize=font_size)

    for plot_config in plot_configs:
        sns.regplot(x=nrows/1e6, y=plot_config['runtimes'], ci=None, color=plot_config["color"], order = 1, marker = plot_config["marker"], label = plot_config["label"],
                scatter_kws={'s':plot_config["s"], 'linewidth':plot_config["lw"]}, 
                line_kws = {'linewidth':plot_config["lw"], 'linestyle': plot_config["ls"]})

    plt.xlim(1.8, 10.2)
    lgnd = plt.legend(frameon=True, prop={'size': font_size})
    plt.show()
    plt.savefig(figaddr) # 

    print (">>>", f"BoXHED vs Blackboost runtime was plotted and saved as {figaddr}.", end="\n\n")


def section_5_train_boxhed_on_MIMIC_IV_iV(training_data):
    nthread = 20

    boxhed_ = boxhed(nthread = nthread)
    X_post  = boxhed_.preprocess(data = training_data, is_cat = list(range(3, 11+1)),
                                            num_quantiles = 256, weighted = False, nthread = nthread)

    boxhed_.fit(X_post['X'], X_post['delta'], X_post['w'])

    return boxhed_


# This function plots the variable importances in BoXHED
def section_5_plot_boxhed_var_imps(varimps, fig_addr, top_k=10):
    def plot_var_imps(vars, imps):
        vars = vars[:top_k]
        imps = imps[:top_k]

        font_size = 40
        _, axis = plt.subplots(figsize=(20,12), dpi=100)
        plt.xticks(fontsize= font_size-6)
        plt.yticks(fontsize= font_size)
        plt.title("Relative Variable Importance", fontsize=font_size)
        plt.bar(vars, imps, color='steelblue')
        plt.xticks(rotation = -90, weight='bold')
        labels = axis.set_xticklabels(vars)
        for label in labels:
            label.set_y(label.get_position()[1]+0.967)
        plt.savefig(fig_addr)

        print (">>>", f"BoXHED relative variable importances for invasive ventilation in MIMIC IV were plotted and saved at {fig_addr}.", end="\n\n")

    # converting the dictionary of variable:importance to two separate lists: variables, importances
    vars, imps    = [np.array(arr) for arr in zip(*list(varimps.items()))]
    srtd_imp_idxs = imps.argsort()[::-1]
    vars          = vars[srtd_imp_idxs]
    imps          = imps[srtd_imp_idxs]
    imps          = imps/imps.max()

    transform = {
        'Fraction inspired oxygen': 'Fraction inspired O2',
        'Glascow coma scale total': 'Glascow coma scale',
        'Oxygen saturation':        'O2 saturation',
        'Heart Rate':               'Heart rate',
    }

    vars = [transform[var] if var in transform else var for var in vars]

    plot_var_imps(vars, imps)


def section_5_plot_hazard_over_time(times, hazards, fname):
    font_size = 40
    ms   = 25
    lw   = 8

    fig, ax = plt.subplots(figsize=(20, 12), dpi=100)

    ax.spines["left"].set_position(("data", 24))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set(lw=lw)
    ax.spines["bottom"].set(lw=lw)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(1, 0, ">k", ms=ms, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(24, 1, "^k", ms=ms, transform=ax.get_xaxis_transform(), clip_on=False)

    ax.text(1.03, -0.03*hazards.max(), s=r"$t$ $(hour)$", fontsize=font_size+20, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(-0.11, 1.05*hazards.max(), s=r"$\hat\lambda(t, X(t))$", fontsize=font_size+20, transform=ax.get_yaxis_transform(), clip_on=False)

    plt.xlim([times.min(), times.max()+2])
    plt.ylim([0, hazards.max()])

    ax.plot(times, hazards, lw=4, color='k')
    
    xticks_ = [24*i for i in range(1, int(times.max()/24)+1)]
    plt.xticks(xticks_, xticks_, fontsize= font_size)
    plt.yticks(fontsize= font_size)
    ax.tick_params(axis='both', which='major', pad=15)
    plt.tight_layout()
    fig.savefig(fname)

    print (">>>", f"Sample patient hazard trajectory for the invasive ventilation in MIMIC IV was plotted and saved at {fname}.", end="\n\n")


if __name__=="__main__":
    # ================================================================================================================ #
    # ================================================================================================================ #
    # ================================================================================================================ #
                                                        # SECTION 4
    # ================================================================================================================ #
    # ================================================================================================================ #
    # ================================================================================================================ #
    
    print_in_terminal ("SECTION 4")

    train_data, test_X = [pd.read_csv(os.path.join('./data/', fname)) for fname in ['training.csv', 'testing.csv']]

    print ('INPUT DATA:')
    print (train_data.head(3), end='\n\n')

    boxhed_ = boxhed()  # Create an instance of BoXHED

    X_post = boxhed_.preprocess(data = train_data, 
                                num_quantiles = 256, 
                                #split_vals   = {"t": [0.2, 0.4, 0.6, 0.8], "X_2": [0, 0.4, 0.9]},
                                weighted      = False, 
                                nthread       = -1)  # Preprocess the training data


    #The set of hyperparameters to cross-validate on (more trees and/or deeper trees may be needed for other datasets)
    param_grid = {
            'max_depth':    [1, 2, 3, 4, 5],                 #a tree of depth k has 2^k leaf nodes
            'n_estimators': [50, 100, 150, 200, 250, 300],   #number of trees in the boosted ensemble
            'eta':          [0.1]                            #stepsize shrinkage, usually held fixed at a small number
        }

    cv_rslts = cv(param_grid, 
                X_post,
                5,                   # num folds
                [-1],                # training with CPU cores
                nthread = -1,        #-1 means use all available CPU threads if training with CPU 
                models_per_gpu = 5)  #the optimal number to use for a specific problem will require some exploration


    #print the K-fold means and standard errors of the log-likelihood values
    import numpy as np
    nrow, ncol = len(param_grid['max_depth']), len(param_grid['n_estimators'])
    print('K-fold mean log-likelihood:')
    print( np.around( cv_rslts['score_mean'].reshape(nrow, ncol), 2), end='\n\n')
    print('K-fold standard error:')
    print( np.around( cv_rslts['score_ste'].reshape(nrow, ncol), 2),  end='\n\n')

    #user-defined measure of model complexity
    def model_complexity(max_depth, n_estimators):
        from math import log2
        return log2(n_estimators) + max_depth

    best_params, params_within_1se = best_param_1se_rule(cv_rslts,
                                                        model_complexity,
                                                        bounded_search=True)   #default is True

    print ('BEST PARAMS:')
    print (best_params, end='\n\n') # Print chosen hyperparameters

    best_params['gpu_id'] = -1            # Use CPU cores to fit BoXHED
    best_params['nthread'] = -1           # If fitting with CPU, the number of threads to use (-1 means use all available threads) 
    boxhed_.set_params(**best_params)

    boxhed_.fit(X_post['X'], X_post['delta'], X_post['w'])

    print ("VARIABLE IMPORTANCES:")
    for k in sorted (boxhed_.VarImps.keys()):
        print (k+':'+(5-len(k))*' '+str(round(boxhed_.VarImps[k], 2)))
    print (end="\n")

    print ("TIME SPLITS:")
    print (boxhed_.time_splits, end="\n\n") # The set of time values where tree splits are made 

    
    section_4_plot_hazard_estimations(boxhed_, test_X, './rslts/hazard.jpeg')

    # Create a dataframe df_surv where each row (t, x_0, x_1, ...) is a point we wish
    # to estimate the value of the survivor curve at
    t           = [t/100 for t in range(0, 100)]
    df_surv      = pd.concat([test_X.loc[2000].to_frame().T]*len(t)).reset_index(drop=True)
    df_surv['t'] = t

    print ("SURVIVOR CURVE DATA FRAME:")
    print (df_surv.head(), end="\n\n") # The set of time values where tree splits are made 

    section_4_plot_survivor(boxhed_, df_surv, './rslts/survivor.jpeg')

    
    # ================================================================================================================ #
    # ================================================================================================================ #
    # ================================================================================================================ #
                                                    # SECTION 5 - SCALABILITY
    # ================================================================================================================ #
    # ================================================================================================================ #
    # ================================================================================================================ #
    
    print_in_terminal ("SECTION 5 - SCALABILITY")

    # we have included the runtime results presented in the paper in the following files.
    boxhed_runtimes_file       = "./rslts/boxhed_runtime.txt"
    blackboost_runtimes_file   = "./rslts/blackboost_runtime.txt"
    scalability_plot_addr      = "./rslts/b2vsblackboost.jpeg"

    # to redo the scalability analysis on your system, you may uncomment the section below to download the data, time both boxhed and blackboost, and plot the new runtimes
    
    redo_scalability_analysis  = False
    if not redo_scalability_analysis:
        print ("NOT redoing BoXHED2.0 vs Blackboost runtime analysis."
               " Only plotting the numbers obtained by the authors on their machine.\n"
               "To redo the analysis on your system, negate the boolean *redo_scalability_analysis*"
               " on line 395 of main.py.\n"
               f"This will overwrite files {boxhed_runtimes_file} and {blackboost_runtimes_file}"
               " which are already included.", end="\n\n")
    else:
        train_data_addr            = "./data/boxhed_scalability_data.csv.gz"
        blackboost_runtimes_script = "blackboost_runtime.R"
 
        section_5_download_train_data(train_data_addr)
        section_5_boxhed_runtime(train_data_addr, boxhed_runtimes_file)
        section_5_blackboost_runtime(blackboost_runtimes_script, train_data_addr, blackboost_runtimes_file)
    
    section_5_plot_boxhed_blackboost_runtimes (boxhed_runtimes_file, blackboost_runtimes_file, scalability_plot_addr)
    
    # ================================================================================================================ #
    # ================================================================================================================ #
    # ================================================================================================================ #
                                        # SECTION 5 - Invasive Ventilation in MIMIC IV
    # ================================================================================================================ #
    # ================================================================================================================ #
    # ================================================================================================================ #
    print_in_terminal ("SECTION 5 - Invasive Ventilation in MIMIC IV")

    var_imps_addr             = "./rslts/boxhed_var_imps.pkl"
    sample_patient_hzrds_addr = "./rslts/sample_patient_hzrds.pkl"

    redo_invasive_ventilation_analysis  = True
    if not redo_invasive_ventilation_analysis:
        print( "The MIMIC IV v1.0 dataset cannot be shared due to being a restricted access resource.\n"
               "For gaining access please refer to their official website: https://physionet.org/content/mimiciv/1.0/.\n"
               "In case you have/gain access to the dataset, you can flip the boolean *redo_invasive_ventilation_analysis*\n"
               "on line 425 of main.py to redo the invasive-ventilation analysis presented in the paper.\n"
               "In that case, you may go to lines 441-443 of main.py to set the addresses that MIMIC IV preprocessor\n"
               "needs in order to function properly, one of which is the address to MIMIC IV v1.0 files.\n")

        print(f"We have saved two files which were presented in the paper: BoXHED variable importances ({var_imps_addr})\n"
              f"and a sample patient hazard trajectory ({sample_patient_hzrds_addr}) for the first 100 hours of their stay\n"
               "(between the 24th hour and 100th in their stay.) If you redo the MIMIC IV invasive-ventilation analysis\n"
               "as explained above, the mentioned results will be overwritten.\n")
    else:

        from MIMIC_IV_extractor.extract_mimic_iv import extract_mimic_iv
        mimic4_path           = "/data/datasets/MIMICIV/physionet.org/files/mimiciv/1.0"       # the address to MIMIC IV v1.0 files
        tmp_path              = "./tmp"       # a directory where intermediate files in the preprocessing can be saved.
        data_path             = "./data"      # the directory where the preprocessed data will be saved
        data_addr             = os.path.join(data_path, "mimic_iv_{mode}.csv")
        extract_mimic_iv(mimic4_path, tmp_path, data_path)
        train_data, test_data = [pd.read_csv(data_addr.format(mode=mode)) for mode in ["train", "test"]]

        boxhed_ = section_5_train_boxhed_on_MIMIC_IV_iV(train_data)
        boxhed_varimps = boxhed_.VarImps
        dump_pickle(boxhed_varimps, var_imps_addr)
    
        sample_patient = test_data[(test_data['ID']==39488188) & (test_data['t_start']<200)]
        sample_patient_hzrds = boxhed_.hazard(sample_patient.drop(columns=["ID", "t_end", "delta", "Y"]))
        dump_pickle([sample_patient['t_start'].values, sample_patient_hzrds], sample_patient_hzrds_addr)
    
    boxhed_varimps = load_pickle(var_imps_addr)
    sample_patient_time_stamps, sample_patient_hzrds = load_pickle(sample_patient_hzrds_addr)
    section_5_plot_boxhed_var_imps(boxhed_varimps, "./rslts/BoXHED_VarImps.jpg")
    section_5_plot_hazard_over_time(sample_patient_time_stamps, sample_patient_hzrds, "./rslts/sample_trajectory.jpeg")