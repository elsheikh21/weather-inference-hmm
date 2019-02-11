import numpy as np

from hmm_inference import (backward_algorithm, fixed_lag_smoothing,
                           forward_algorithm, forward_backward_algorithm,
                           improved_forward_backward)
from initialize_hmm import init
from sampling import sampling
from utils import write_to_file, dict_to_matrix


# 1. Instantiate Hidden Markov Model[HMM]
(states, observations, transition_model, observation_model,
 observation_true_model, observation_false_model) = init()

# values of the initialization are saved with no labels
# so just map them from method params in same order
write_to_file((states, observations, transition_model, observation_model,
               observation_true_model, observation_false_model),
              "Markov Model initialization.txt")

# 2. Get samples from our model
iterations = 15
sequence_length = 20
sampling(iterations, sequence_length, states,
         observations, transition_model, observation_model)

# 3. Implement forward-backward algorithm
start_distribution_probability = np.array([0.5, 0.5])
forward_list, last_forward_list = forward_algorithm(iterations,
                                                    transition_model,
                                                    observation_true_model,
                                                    observation_false_model,
                                                    start_distribution_probability)

write_to_file(forward_list, 'Forward algorithm output.txt')
write_to_file(last_forward_list,
              'Forward algorithm output(Last entry per sequence).txt')

backward_list = backward_algorithm(iterations, sequence_length,
                                   transition_model,
                                   observation_true_model,
                                   observation_false_model)

write_to_file(backward_list, "Backward algorithm output.txt")

forward_backward_list = forward_backward_algorithm(
    forward_list, backward_list, start_distribution_probability)

write_to_file(forward_backward_list, "Forward Backward algorithm output.txt")

# Changing structure of the parameters to be matrices
trans_mdl = dict_to_matrix(transition_model)
obs_true_mdl = dict_to_matrix(observation_true_model)
obs_false_mdl = dict_to_matrix(observation_false_model)
forward_lst = np.array(forward_list)

time_space_indpndt_mem = improved_forward_backward(iterations=iterations,
                                                   last_forward_storage=last_forward_list,
                                                   transition_model=trans_mdl,
                                                   true_observation_model=obs_true_mdl,
                                                   false_observation_model=obs_false_mdl)

write_to_file(time_space_indpndt_mem,
              "Improved Forward Backward algorithm output.txt")

fixed_lag_smoothing = fixed_lag_smoothing(iterations,
                                          start_distribution_probability,
                                          obs_true_mdl,
                                          obs_false_mdl,
                                          10, trans_mdl, forward_lst)

write_to_file(fixed_lag_smoothing, "Fixed lag smoothing algorithm output.txt")
