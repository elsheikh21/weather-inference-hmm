import os
import numpy as np


def sampling(iterations, sequence_length, states, observations,
             transition_model, observation_model):
    '''
    Given, the model prior probabilities, transition probabilities,
    observations, required to generate a number of samples, to be
    able to evaluate the sequence.
    We will develop a Q_i = {q_1, q_2, ...., Q_T}, i = 15 sequences,
    and T = 20 days, as per a sequence
    '''
    print('[INFO] Sampling in progress')
    directory = os.path.join(os.getcwd(), 'Samples')

    if not os.path.exists(directory):
        os.makedirs(directory)

    for iteration in range(iterations):
        filename = os.path.join(
            directory, 'sample{}.txt'.format(str(iteration+1)))
        sample = open(filename, 'w')
        # States are given, to choose one from [S] randomly,
        # and initiate the coming days of this sequence
        previous_state = np.random.choice(a=states)
        for counter in range(sequence_length):
            # Observations and probabilities associated with each probability
            # to choose one from [O] randomly
            umbrella = np.random.choice(
                a=observations, p=observation_model.get(previous_state))
            row = str(counter+1) + ' ' + previous_state + ' ' + umbrella
            sample.write(row + '\n')
            # Update the previous state with the current state, based on
            # the given states and their transition model,
            # as we are transitioning from S_t to S_t+1
            previous_state = np.random.choice(
                a=states, p=transition_model.get(previous_state))
        sample.close()
    print('[INFO] Created {} samples, each of {} days. Sampling is saved.\n'.format(
        iterations, sequence_length))
