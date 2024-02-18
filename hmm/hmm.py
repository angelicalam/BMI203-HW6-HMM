import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        # Check input shapes
        if prior_p.shape != hidden_states.shape:
            raise ValueError("incorrect number of prior probabilities provided")
        if transition_p.shape != (hidden_states.shape[0], hidden_states.shape[0]):
            raise ValueError("transition probabilities matrix has the wrong dimensions")
        if emission_p.shape != (hidden_states.shape[0], observation_states.shape[0]):
            raise ValueError("emission probabilities matrix has the wrong dimensions")
        # Check that probabilities sum to 1
        if not np.all(transition_p.sum(axis=1)==1):
            raise ValueError("transition probabilities leaving each hidden state must sum to 1")
        if not np.all(emission_p.sum(axis=1)==1):
            raise ValueError("emission probabilities for each hidden state must sum to 1")
        if not np.isclose(prior_p.sum(), 1.0):
            raise ValueError("prior probabilities must sum to 1")
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p   # rows: from hidden state, columns: to hidden state
        self.emission_p = emission_p     # rows are hidden states, columns are observation states


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        # Step 1. Initialize variables
        # Forward path probabilities for all hidden states (columns) 
        # for all input sequence observation states
        self.forward_alpha = np.zeros((len(input_observation_states), len(self.hidden_states)))
        
        # Calculate t=1 forward path probability using prior probabilities:
        # alpha_t=1 = P(hidden state i | start) x P(first observable | hidden state i)
        o = self.observation_states_dict[input_observation_states[0]]   # one-hot encoded first observable
        self.forward_alpha[0] = [self.prior_p[i] * self.emission_p[i][o] for i in self.hidden_states_dict.keys()]
       
        # Step 2. Calculate probabilities
        # alpha_t(j) = sum over hidden states i {
        #                alpha_t-1(i) x transition_p(i to j) X emission_p(j to observation state t) }
        for t in range(1, len(input_observation_states)):
            for j in self.hidden_states_dict.keys():
                ot = self.observation_states_dict[input_observation_states[t]]  # observation state t
                self.forward_alpha[t][j] = np.sum([ 
                                            self.forward_alpha[t-1][i] * self.transition_p[i][j] * self.emission_p[j][ot]
                                            for i in self.hidden_states_dict.keys() ])
        print(self.forward_alpha)
        # Step 3. Return final probability
        # P(observation state sequence | HMM) = sum of forward path probabilities at t=T over all hidden states
        likelihood = np.sum(self.forward_alpha[len(input_observation_states)-1])
        return likelihood


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """
        # Step 1. Initialize variables  
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states), dtype=int)
        # Stores probability maximizing predecessor states for traceback
        best_path_table = np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype=int)
        # Convenience variable for getting max likelihood hidden states
        hidden_states_j = np.sort(list(self.hidden_states_dict.keys()))
        
        # # Calculate initial hidden state probabilities using prior_p:
        # s_(k,i) = P(first observable | hidden state k) x prior_p(k)
        # Use log probabilities to avoid underflow.
        o = self.observation_states_dict[decode_observation_states[0]]   # one-hot encoded first observable
        with np.errstate(divide='ignore'):
            viterbi_table[0] = [np.log(self.emission_p[k][o] * self.prior_p[k]) for k in hidden_states_j]
        # Log of zero is undefined, but np.log(0) = -inf and gives a divide by zero warning. 
        # For the purposes of the viterbi algorithm, calculated probabilities should be the most low
        # when encountering emission/transition probabilities of zero. 
        # Leaving these probabilities as -inf effectively does this.
        
        # Step 2. Calculate Probabilities and update best_path.
        # s_(k,i) = P(observable | hidden state k) x max_j{ s_(j,i-1) * P(k|j) }
        # where max_j is computed for all hidden states j.
        # Use log probabilities to avoid underflow.
        for i in range(1, len(decode_observation_states)):
            for k in self.hidden_states_dict.keys():
                oi = self.observation_states_dict[decode_observation_states[i]]  # observation state i
                # find the max log likelihood hidden state j
                with np.errstate(divide='ignore'):
                    joint_p = [viterbi_table[i-1][j] + np.log(self.transition_p[j][k]) for j in hidden_states_j]
                # Since hidden_states_j is sorted and consists of [0, 1, ..., len(hidden_states)], 
                # argmax is returning the max log likelihood hidden state j
                best_path_table[i-1][k] = np.argmax(joint_p)
                with np.errstate(divide='ignore'):
                    viterbi_table[i][k] = np.log(self.emission_p[k][oi]) + np.max(joint_p)

        # Step 3. Traceback
        # Start with the greatest final probability
        best_path[len(decode_observation_states)-1] = np.argmax(viterbi_table[len(decode_observation_states)-1])
        # Iterate backwards
        for i in reversed(range(len(decode_observation_states)-1)):
            # Pick the predecessor state that resulted in the current "winning" probability
            best_path[i] = best_path_table[i][best_path[i+1]]
        
        # Step 4. Return best hidden state sequence 
        # Convert sequence of hidden state indices to their respective states
        best_path = [self.hidden_states_dict[j] for j in best_path]
        return np.array(best_path)
        