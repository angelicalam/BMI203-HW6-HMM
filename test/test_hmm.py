import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    # Create instance of HMM
    HMM = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], 
                            mini_hmm['prior_p'], mini_hmm['transition_p'], mini_hmm['emission_p'])
    # Run Forward algorithm to 
    # calculate the probability of sequence mini_input given mini_hmm 
    mini_in_p = HMM.forward(mini_input['observation_state_sequence'])
    # allow leeway for python float imprecision
    assert np.isclose(mini_in_p, 0.03506441115), "wrong probability returned"
    
    # Run Viterbi algorithm to 
    # get the most probable hidden state sequence given mini_input
    mini_hidden_seq = HMM.viterbi(mini_input['observation_state_sequence'])
    assert len(mini_hidden_seq) == len(mini_input['observation_state_sequence']), "wrong number of states returned"
    assert np.array_equal(mini_hidden_seq, mini_input['best_hidden_state_sequence']), "wrong order returned"
    
    # Checking edge cases
    # Transition/emission probabilities that do not add up to 1 should throw an error.
    with pytest.raises(ValueError) as excinfo:
        model = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], 
                            mini_hmm['prior_p']-0.1, mini_hmm['transition_p'], mini_hmm['emission_p'])
    with pytest.raises(ValueError) as excinfo:
        model = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], 
                            mini_hmm['prior_p'], mini_hmm['transition_p']-0.1, mini_hmm['emission_p'])
    with pytest.raises(ValueError) as excinfo:
        model = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], 
                            mini_hmm['prior_p'], mini_hmm['transition_p'], mini_hmm['emission_p']-0.1)
        
    # Bad input shapes should throw an error.
    with pytest.raises(ValueError) as excinfo:
        model = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], 
                            np.array([.2,.2,.2,.2,.2]), mini_hmm['transition_p'], mini_hmm['emission_p'])
    with pytest.raises(ValueError) as excinfo:
        model = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], 
                            mini_hmm['prior_p'], np.array([[.2,.2,.6],[.6,.2,.2]]), mini_hmm['emission_p'])
    with pytest.raises(ValueError) as excinfo:
        model = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'], 
                            mini_hmm['prior_p'], mini_hmm['transition_p'], np.array([[.2,.2,.6],[.6,.2,.2]]))
        
    # Zero-probability transitions and emissions.
    model = HiddenMarkovModel(mini_hmm['observation_states'], mini_hmm['hidden_states'],
                              np.array([.5, .5]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]) )
    likelihood = model.forward(np.array(['rainy', 'sunny']))
    # If hot can only transition to hot, and cold can only transition to cold, and
    # if hot can only manifest sunny, and cold can only manifest rainy,
    # then the likelihood of rainy, sunny is zero.
    assert likelihood == 0
    # Test that viterbi() can also handle zero-probability transitions and emissions.
    # Log of zero is undefined, but np.log(0) = -inf. 
    # For the purposes of the viterbi algorithm, calculated probabilities should be the most low
    # when encountering emission/transition probabilities of zero. 
    # Leaving these probabilities as -inf effectively does this.
    path = model.viterbi(np.array(['rainy', 'rainy']))
    assert np.array_equal(path, ['cold', 'cold']), "wrong order returned"
    
    # Test that viterbi() can handle very long observation sequences.
    mini_hidden_seq = HMM.viterbi(np.array(mini_input['observation_state_sequence'].tolist() * 50))
    assert np.array_equal(mini_hidden_seq, 
                          np.array(mini_input['best_hidden_state_sequence'].tolist() * 50)), "wrong order returned"


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    # Create instance of HMM
    HMM = HiddenMarkovModel(full_hmm['observation_states'], full_hmm['hidden_states'], 
                            full_hmm['prior_p'], full_hmm['transition_p'], full_hmm['emission_p'])
    # Run Forward algorithm to 
    # calculate the probability of sequence mini_input given mini_hmm 
    full_in_p = HMM.forward(full_input['observation_state_sequence'])
    assert (full_in_p > 0) & (full_in_p < 1)
    
    # Run Viterbi algorithm to 
    # get the most probable hidden state sequence given mini_input
    full_hidden_seq = HMM.viterbi(full_input['observation_state_sequence'])
    assert len(full_hidden_seq) == len(full_input['observation_state_sequence']), "wrong number of states returned"
    assert np.array_equal(full_hidden_seq, full_input['best_hidden_state_sequence']), "wrong order returned"




