import os
import sys
import random

from math import exp

from pyneat import pyneat

# CLI Arguments
load_save = False
for arg in sys.argv:
    if arg == '--load':
        load_save = True

# Experiment Params
input_nodes = 2
output_nodes = 2

population_size = 1000
num_generations = 100

# Population cutoff percentage
cutoff = 0.15

# Percentage chance to mutate:
structure_mutation = 0.2
weight_mutation = 0.8

# Constants used in speciation
c1 = 1.0
c2 = 1.0
c3 = 0.4

# Speciation threshold
sigma_t = 3.0

# Define a fitness function a genome can use to evaluate itself
# Note: remember to keep this in scope of a fork.
def fitness_func(self):
    global input_nodes

    # Remember this isn't backprop, the batch
    # is just for evaluating loss
    batch_size = 32

    # Create a batch of random inputs
    vec_in = []
    for _ in range(batch_size):
        x = []
        for _ in range(input_nodes):
            x.append(float(random.randint(0,1)))
        vec_in.append(x)

    # Calculate network outputs
    vec_out = [self.network.forward(x) for x in vec_in]

    # This network will evolve to detect if all inputs are high
    expected = []
    for x in vec_in:
        a = [1.0,0.0] if sum(x) == input_nodes else [0.0,1.0]
        expected.append(a)

    # Fitness = the inverse loss
    mse = []
    for y, y_ in zip(vec_out, expected):
        for i in range(len(y)):
            error = (y[i] - y_[i]) ** 2
            mse.append(error)
    self.fitness = 1 - (sum(mse) / len(mse))

    # In this experiment, we have a quantifiable optimum,
    # thus we should save this particular genome.
    if self.fitness > 0.98:
        print('\n~ found optimal network: {} ~'.format(self.fitness))

        # Create and run 5 tests
        tests = [[random.uniform(0.0,1.0) for _ in range(input_nodes)] for _ in range(5)]
        for x in tests:

            # In this case we reset the network to ignore recurrent connections,
            # we haven't been using them during evolution, thus we won't here.
            self.network.reset()
            y = self.network.forward(x)

            # Simplify the log information
            input_text = 'all active' if sum(x) == input_nodes else 'not active'
            output_text = str([round(x,3) for x in y])[1:-1]
            print("{}\t{}".format(input_text, output_text))

            # Save the genome information
            self.save('saves/top.save')

        # Signal network termination, clean up threads
        return True, self

    # Default returns a non-Terminating signal
    # with the modified object.
    return False, self

# If we haven't already evolved a network
if load_save == False:

    # Make sure we are fork protected
    if __name__ == '__main__':

        # Create the gene pool
        gene_pool = pyneat.GenePool(population_size,
                            num_generations,
                            cutoff,
                            [c1,c2,c3],
                            sigma_t,
                            logging=1,
                            num_threads=8)

        # Initialise the gene pool
        gene_pool.init(input_nodes, output_nodes, fitness_func)

        # Run an evolutionary period, specifying that we have defined
        # a signal to save in the fitness function
        gene_pool.evolve(weight_mutation, structure_mutation, saver=pyneat.Saver.USER_DEFINED)

# If we have already evolved a network
else:
    # Load the genome, extract the network
    net = pyneat.Genome().load('saves/top.save').build().network

    # Run the same tests shown in the fitness function
    tests = [[random.uniform(0.0,1.0) for _ in range(input_nodes)] for _ in range(5)]
    for x in tests:
        # Propagate input throughout the network
        net.reset()
        y = net.forward(x)

        # Simplify the log information
        input_text = 'all active' if sum(x) == input_nodes else 'not active'
        output_text = str([round(x,3) for x in y])[1:-1]
        print("{}\t{}".format(input_text, output_text))