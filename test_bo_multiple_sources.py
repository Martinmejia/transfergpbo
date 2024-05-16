import numpy as np
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import pickle

from emukit.core import ParameterSpace, ContinuousParameter

from transfergpbo.benchmarks import hartmann
from transfergpbo import experiment, experiment_1
from transfergpbo import parameters

import warnings
warnings.filterwarnings("ignore")

'''# Techniques available to calculate: 
GPBO - GP_based BO with no source data.

MHGP - Mean Hierarchical BO. The pretrained source transfers information to the target using only the posterior mean to the source as the prior mean for the target.

SHGP - Sequential Hierarchical GP. Asymmetric transfer learning is where all task data influence both source and target model parameters. 
       Target data accumulates during BO while the source remains constant => Bayesian modular approach where target data do not influence the source model.
       The source model parameters do not depend on the target data, but the source posterior does.

BHGP - The source posterior is independent of the target data. This model trains a target GP with the prior mean function sampled from the source GP. 
       The posterior is averaged over all the samples of the source GP. The resulting sample averaged distribution is also a GP.       
'''

technique_list = ['SHGP', 'BHGP', 'MHGP', 'GPBO']
source_points_list = [1, 60, 120, 180]

def run_bo_multiple_sources(technique, source_points):
    """
    Run Bayesian Optimization for a given technique and collect results.

    Parameters:
    - technique (str): The technique to use for Bayesian Optimization.

    Returns:
    Tuple of NumPy arrays:
    - output_mean (np.ndarray): Mean regret over iterations.
    - output_std_error (np.ndarray): Standard error of regret over iterations.
    """

    parameters.parameters['technique'] = technique
    parameters.parameters['benchmark']['num_source_points'] = [source_points]
    output = collections.defaultdict(list)

    try:
        for i in tqdm(range(50), desc=f'Running {technique} with {source_points} source points'):
            np.random.seed(i)
            output['iter'].append(i)
            output['output_tuple_regret'].append(experiment_1.run_experiment(parameters.parameters))


        regret_array     = np.array(output['output_tuple_regret'])
        output_mean      = np.mean(regret_array, dtype=np.float64, axis=0)
        output_std_error = np.std(regret_array, dtype=np.float64, axis=0) / np.sqrt(regret_array.shape[0])

        return regret_array, output_mean, output_std_error
    
    except:
        regret_array     = np.zeros((50,30))
        output_mean      = np.zeros((30,))
        output_std_error = np.zeros((30,))

        return regret_array, output_mean, output_std_error




def plot_means_and_errors_sp(result_dict, acquisition_function):
    """
    Plot means and standard errors for different techniques.

    Parameters:
    - result_dict (dict): A dictionary containing mean and standard error arrays for each technique.

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))

    for key, values in result_dict.items():
        if 'mean' in key:
            technique_name = key.split('_')[0]
            num_source = key.split('_')[-1]
            mean_array = values
            stderr_array = result_dict.get(f'{technique_name}_stderr_{num_source}', np.zeros_like(mean_array))
            
            x_values = np.arange(1,len(mean_array)+1)
            
            sns.lineplot(x=x_values, y=mean_array, label=f'{technique_name}_{num_source}', marker='o')
            plt.fill_between(x=x_values, y1=mean_array - stderr_array, y2=mean_array + stderr_array, alpha=0.25)

    plt.yscale('log')  
    plt.xlabel('Iteration')
    plt.ylabel('Regret (log scale)')
    plt.title(f'Means with Standard Errors Hartmann3d with {acquisition_function} acquisition function')
    plt.legend()
    plt.savefig('./EI_acc/best_models_multiple_sources.png', dpi=600)
    plt.close()


file_1 = './EI_acc/results_dict_best_multiple_sources.pkl'
file_2 = './EI_acc/regret_array_best_multiple_sources.pkl'

if __name__ == '__main__':

    result_dict = {}
    array_dict = {}
    acquisition_function = 'EI'

    for name in tqdm(technique_list):
        for num in source_points_list:
            regret, mean, stderr = run_bo_multiple_sources(name, num)
            array_dict[f'{name}_regret_{num}'] = regret
            result_dict[f'{name}_mean_{num}'] = mean
            result_dict[f'{name}_stderr_{num}'] = stderr



    with open(file_1, 'wb') as file:
        pickle.dump(result_dict, file)

    with open(file_2, 'wb') as f:
        pickle.dump(array_dict, f)

        plot_means_and_errors_sp(result_dict, acquisition_function)
