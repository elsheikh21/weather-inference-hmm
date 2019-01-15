import numpy as np
from initialize_hmm import init
from sampling import sampling
from hmm_inference import (
    forward_algorithm, backward_algorithm, forward_backward_algorithm)


# 1. Instantiate Hidden Markov Model[HMM]
(states, observations, transition_model, observation_model,
 observation_true_model, observation_false_model) = init()

# 2. Get samples from our model
iterations = 15
sequence_length = 20
sampling(iterations, sequence_length, states,
         observations, transition_model, observation_model)

# 3. Implement forward-backward algorithm
start_distribution_probability = np.array([0.5, 0.5])
forward_list = forward_algorithm(iterations, transition_model,
                                 observation_true_model,
                                 observation_false_model,
                                 start_distribution_probability)

backward_list = backward_algorithm(iterations, sequence_length,
                                   transition_model,
                                   observation_true_model,
                                   observation_false_model)

forward_backward_list = forward_backward_algorithm(
    forward_list, backward_list, start_distribution_probability)
