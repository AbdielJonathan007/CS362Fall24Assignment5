

import random
import argparse
import codecs
import os
import numpy
import numpy as np


# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}, start_probability={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions
        self.start_probability = start_probability # Add these one for initial statees

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        #Reading from transition basename.trans
        with open(f"{basename}.trans", 'r') as file:
            for line in file:
                # remove leading whitespace
                line = line.strip()
                # '#' Denotes a starting state
                if line.startswith("#"):
                    parts = line[1:].strip().split()
                    state = parts[0]
                    probability = float(parts[1])
                    self.start_probability[state] = probability
                elif line:
                    parts = line.split()
                    state_from = parts[0]
                    state_to = parts[1]
                    probability = float(parts[2])

                    # Initialize inner dictionary if it doesn't exist // Double check that se e mesmo necessario
                    if state_from not in self.transitions:
                        self.transitions[state_from] = {}
                    self.transitions[state_from][state_to] = probability

        with open(f"{basename}.emit", 'r') as file:
            for line in file:
                line = line.strip()
                parts =line.split()
                state = parts[0]
                output = parts[1]
                probability = float(parts[2])

                # Initialize inner dictionary if it doesn't exist
                if state not in self.emissions:
                    self.emissions[state] = {}
                self.emissions[state][output] = probability

    # used chatgpt as a reference to this part
   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        # select an initial state from our states
        # for i in nIterations:
            # use P(et|Xt) to generate an observation
            # Use P(xt+1 | Xt) to generate a next state
        # return n
        # Choosing the initial state
        initial_state = list(self.start_probability.keys())
        start_probalility = list(self.start_probability.values())
        current_state = numpy.random.choice(initial_state, p=start_probalility)

        # Initialize the sequences
        state_sequence = [current_state] # Starting with the initial state
        output_sequence = []

        for _ in range(n):
            emissions = list(self.emissions[current_state].keys())
            emission_probs = list(self.emissions[current_state].values())
            output_sequence.append(numpy.random.choice(emissions, p= emission_probs))

            # Choosing the next state based on the current state's transition probabilities

            next_states = list(self.transitions[current_state].keys()) # looks incorrect
            transition_probs = list(self.transitions[current_state].values())
            current_state = numpy.random.choice(next_states, p=transition_probs)

            # Appending the new state to the state sequence
            state_sequence.append(current_state)
        return Sequence(state_sequence, output_sequence)


    #Pseudo-code on slides
    def forward(self, sequence):
        T = len(sequence)
        states = list(self.transitions.keys())

        # Initialize a matrix to store forward probabilities
        forward_prob = np.zeros((T, len(states)))

        for i, state in enumerate(states):
            forward_prob[0, i] = self.start_probability.get(state, 0) * self.emissions[state].get(sequence[0], 0)

        # Recursively calculate probabilities for each time step
        for t in range(1, T):
            for i, state in enumerate(states):
                forward_prob[t, i] = sum(
                    forward_prob[t -1, j] * self.transitions[prev_state].get(state, 0) * self.emissions[state].get(sequence[t], 0)
                    for j, prev_state in enumerate(states)
                )

        final_probability = sum(forward_prob[T -1, :])
        return final_probability

    # used chatgpt as a reference to this part
    #Pseudo-code on slides
    def viterbi(self, sequence):
        T = len(sequence)
        states = list(self.transitions.keys())

        viterbi_prob = np.zeros((T, len(states)))
        backpointer = np.zeros((T, len(states)), dtype=int)

        for i, state in enumerate(states):
            viterbi_prob[0, i] = self.start_probability.get(state, 0) * self.emissions[state].get(sequence[0], 0)

        # Viterbi recursion
        for t in range(1, T):
            for i, state in enumerate(states):
                max_prob, max_state = max(
                    (viterbi_prob[t - 1, j] * self.transitions[prev_state].get(state, 0), j)
                    for j, prev_state in enumerate(states)
                )
                viterbi_prob[t,i] = max_prob * self.emissions[state].get(sequence[t], 0)
                backpointer[t,i] = max_state

        best_path = [0] * T
        best_path[-1] = np.argmax(viterbi_prob[T-1, :])
        for t in range(T-2,-1,-1):
            best_path[t] = backpointer[t+1, best_path[t+1]]

        # Convert state indices back to state names
        best_states = [states[i] for i in best_path]
        return best_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM on different tasks.")
    parser.add_argument("domain", help="Specify the domain, e.g., 'cat' or 'lander'.")
    parser.add_argument("--generate", type=int, help="Generate a random sequence of length n.")
    parser.add_argument("--forward", help="Run the forward algorithm on a sequence file.")
    parser.add_argument("--viterbi", help="Run the Viterbi algorithm on a sequence file.")
    args = parser.parse_args()

    h = HMM()
    h.load(args.domain)

    if args.generate:
        sequence = h.generate(args.generate)
        print(sequence)

    elif args.forward:
        with open(args.forward) as f:
            observations = f.read().strip().split()
        final_state_prob = h.forward(observations)
        print("Final state probability:", final_state_prob)

    elif args.viterbi:
        with open(args.viterbi) as f:
            observations = f.read().strip().split()
        best_path = h.viterbi(observations)
        print("Most likely sequence of states:", ' '.join(best_path))





