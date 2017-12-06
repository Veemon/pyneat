'''
-crossover- TODO

structure -mutations- occur in two ways:

    in -add connection- mutation, a single new connection gene with a random weight is added
    connecting two previously unconnected nodes

    in the -add node- mutation, an existing connection is split and the new node placed where
    the old connection used to be.
        the old connection is disabled
        the new connections are then added
            - the new connection leading into the new node has a weight of 1
            - the new connection leading out of the new node has the weight of the old connection

whenever a new gene appears through structural mutation, a global innovation number is incremented and assigned to that gene

it is possible to ensure that when the same structure arises more than once through independent mutations
in the same generation, each identical mutation is assigned the same innovation number.

when -crossover- occurs, the genes in both genomes with the same innovation numbers are lined up.
these are called matching genes.

genes that do not match are either disjoint or excess, depending on whether they occur within or outside the range of the parents
innovation numbers.

when composing offspring, genes are randomly chosen from either parent at matching genes,
whereas all excess or disjoint genes are always included from the more fit parent.

the number of excess and disjoint genes between a pair of genomes is a natural measure of their compatibility distance.
the more disjoint two genomes are, the less evolutionary history they share, and thus the less compatibile they are.

sigma = c1(E/N) + c2(D/N) + c3*W
sigma is the compatibility difference
c1,c2,c3 are coefficients
E is the number of excess genes
D is the number of disjoint genes
W is the average weight differences of matching genes (including disable genes)
N is the number of genes in the larger genome

this compatibility difference allows for speciation, with a compatibility threshold sigma_t.

in each generation genomes are sequentially placed into species.
each existing species is represented by a random genome inside the species from the previous generation.

a given genome g in the current generation is placed in the first species with which g is compatibile with
the respective genome of that species.
if g is not compatibile with any of the existing species, a new species is created with g as the representative.

explicit fitness sharing is used, where organisms in the same species must share the fitness of their niche.
this is to combat dominant species.

the adjusted fitness for the organism is calculated according to its compatibility distance from every other organism
in the population.

f_i' = f_i / sum( sh( sigma(i,j) ) )
f_i' is the adjusted fitness for organism i
f_i = is the fitness for organism i
sigma(i,j) is the compatibility distance between organisms i and j

sh is the sharing function
    it is set to 0 when the compatibility distance is above the threshold
    otherwise it is set to 1

species then reproduce by first eliminating the lowest performing members from the population.
the entire population is then replaced by the offspring of the remaining organisms in each species.

neat starts out with a uniform population of networks with zero hidden nodes.
all inputs are directly connected to all outputs.
'''




# standard library
import random
import sys

from collections import namedtuple
from enum import Enum
from math import exp

# third-party library
from tabulate import tabulate
import numpy as np




# Helper Abstractions
Neuron = namedtuple('Neuron',
                        ['type', 'id'])

Connection = namedtuple('Connection',
                        ['input', 'output', 'weight', 'enabled', 'innovation'])

class Type(Enum):
    IN = 0
    HIDDEN = 1
    OUT = 2




# Activation Functions
def sigmoid(x):
    return 2 / (1 + exp(-4.9*x)) - 1

def relu(x):
    return x if x > 0 else 0

# Nodes - acts like an I/O processor
class Node():
    def __init__(self, idx, node_type):
        self.id = idx
        self.type = node_type

        self.cache = []
        self.output = 0
        self.ready = False

        self.input_counter = 0
        self.input_nodes = []

        self.weights = []
        self.output_nodes = []

        self.recurrent_out = []

    def detect_recurrence(self):
        # recurrence with output nodes
        if self.type == Type.OUT:

            # we assume that all output nodes must be recurrent
            if len(self.output_nodes) > 0:
                for node in self.output_nodes:
                    node.input_nodes.remove(self)
                    self.recurrent_out.append(node)
                self.output_nodes = []

        # recurrence with hidden nodes
        elif self.type == Type.HIDDEN:

            # check if node is in inputs
            for node1 in self.output_nodes:
                if node1 in self.input_nodes:

                    # check if that node, has an output to this node
                    for node2 in node1.output_nodes:
                        if node2.id == self.id:
                            # remove input node in-connection
                            self.input_nodes.remove(node1)

                            # shuffle output node to recurrent
                            node1.output_nodes.remove(self)
                            node1.recurrent_out.append(self)

        # recurrence with input nodes
        elif self.type == Type.IN:

            # we assume that all input nodes must be recurrent
            if len(self.input_nodes) > 0:
                for node in self.input_nodes:
                    # shuffle output node to recurrent
                    node.output_nodes.remove(self)
                    node.recurrent_out.append(self)

                    # remove node connection
                    self.input_nodes.remove(node)

    def forward(self):
        # if we've received all inputs
        if len(self.cache) >= self.input_counter and self.ready == False:

            # forward our output to all nodes
            output = sigmoid(sum(self.cache))
            for i, node in enumerate(self.output_nodes):
                node.cache.append(output * self.weights[i])
                node.input_counter += 1

            # store the output for recurrent pass
            self.output = output
            self.cache = []

            # signal that this node has no jobs
            self.ready = True
            self.input_counter = 0

    def r_forward(self):
        # forward the last output to recurrent connections
        l = len(self.output_nodes)
        for i, node in enumerate(self.recurrent_out):
            node.cache.append(self.output * self.weights[l + i])

    def reset(self):
        self.cache = []
        self.output = 0
        self.input_counter = 0
        self.ready = False

# Network - builds the routing between nodes
class Network():
    def __init__(self, genome):
        # neurons
        self.nodes = []
        for neuron in genome.neurons:
            self.nodes.append(Node(neuron.id, neuron.type))

        # connections
        self.connections = genome.connections

    def find_node(self, node_id):
        # TODO: sort when copying from genome,
        #       then index into the array/dict instead.
        for node in self.nodes:
            if node.id == node_id:
                return node

    def build(self):
        # build all neuron connections
        for connection in self.connections:
            if connection.enabled == True:
                # find corresponding node
                input_node = self.find_node(connection.input)
                output_node = self.find_node(connection.output)

                # update connection
                input_node.weights.append(connection.weight)
                input_node.output_nodes.append(output_node)
                output_node.input_nodes.append(input_node)

        # after all connections are done, detect cycles
        for node in self.nodes:
            node.detect_recurrence()

    def forward(self, x):
        # get input and output nodes
        input_nodes = []
        output_nodes = []
        for node in self.nodes:
            if node.type == Type.IN:
                input_nodes.append(node)
            elif node.type == Type.OUT:
                output_nodes.append(node)

        # split x to feed to input neurons
        for i, node in enumerate(input_nodes):
            node.cache.append(x[i])

        # begin forward pass 1 - propagation
        length = len(output_nodes)
        while True:

            # forward propagate some nodes
            for node in self.nodes:
                node.forward()

            # check if the output nodes are ready
            num_ready = 0
            for node in output_nodes:
                if node.ready == True:
                    num_ready += 1
            if num_ready == length:
                break

        # reset node state signals
        for node in self.nodes:
            node.ready = False

        # begin forward pass 2 - recurrence
        for node in self.nodes:
            node.r_forward()

        # return output
        return [x.output for x in output_nodes]

    def reset(self):
        for node in self.nodes:
            node.reset()

    def __str__(self):
        # shortcut
        nodes = self.nodes

        # tabulate headers
        headers = ['ID', 'IN', 'OUT', 'R']

        # all node id's
        node_ids = [x.id for x in nodes]

        # all respective node inputs
        node_inputs = []
        for node in nodes:
            inputs = []
            for node_in in node.input_nodes:
                inputs.append(node_in.id)
            node_inputs.append(inputs)

        # all respective node outputs
        node_outputs = []
        for node in nodes:
            outputs = []
            for node_out in node.output_nodes:
                outputs.append(node_out.id)
            node_outputs.append(outputs)

        # all respective node recurrents
        recurrents = []
        for node in nodes:
            recur = []
            for node_rec in node.recurrent_out:
                recur.append(node_rec.id)
            recurrents.append(recur)

        # swizzle into mega list
        swizzle = []
        for i in range(len(nodes)):
            current_node_info = []
            current_node_info.append([node_ids[i]])
            current_node_info.append(node_inputs[i])
            current_node_info.append(node_outputs[i])
            current_node_info.append(recurrents[i])
            swizzle.append(current_node_info)

        return tabulate(swizzle, headers=headers, tablefmt="fancy_grid")




# Global innovation counter
Innovation = 0
Connection_Cache = []
Innovation_Cache = []

# Genome - Network Blueprint
class Genome:
    def __init__(self):
        # information
        self.neurons = []
        self.connections = []

        # measurements
        self.network = 0
        self.fitness = 0
        self.get_fitness =  0

    def init(self, num_inputs, num_outputs, fitness_function):
        global Innovation

        # activate fitness function
        self.get_fitness = fitness_function

        # grow neurons
        for i in range(num_inputs):
            self.neurons.append(Neuron(Type.IN, i))
        for i in range(num_outputs):
            self.neurons.append(Neuron(Type.OUT, num_inputs + i))

        # grow connections
        for i in range(num_inputs):
            for j in range(num_outputs):
                self.connections.append(Connection(i, num_inputs + j,
                                                    random.uniform(-1.0, 1.0),
                                                    True,
                                                    (i*num_outputs)+j))

        # update innovation
        Innovation = num_inputs + num_outputs + 1

        return self

    def find_neuron(self, neuron_id):
        for n in self.neurons:
            if n.id == neuron_id:
                return n

    def find_connection(self, innovation):
        # TODO: a lot of linear searches in the codebase
        for c in self.connections:
            if c.innovation == innovation:
                return c

    def build(self):
        self.network = Network(self)
        self.network.build()

    def evaluate(self):
        self.build()
        self.get_fitness(self)

    def __str__(self):
        headers = ['Nodes', 'Connections']
        connection_str = [str([x.input, x.output])[1:-1] for x in self.connections]
        for i, connection in enumerate(connection_str):
            if self.connections[i].enabled == True:
                connection_str[i] = connection.replace(',', ' ->')
            else:
                connection_str[i] = connection.replace(',', ' -x')

        swizzle = []
        l_node = len(self.neurons)
        l_con = len(connection_str)
        length = l_node if l_node > l_con else l_con

        for i in range(length):
            a = ''
            b = ''

            if i < l_node:
                a = str(self.neurons[i].id)

            if i < l_con:
                b = connection_str[i]

            swizzle.append([a,b])

        return tabulate(swizzle, headers=headers)

# Creates offspring between two genomes
def crossover(g1, g2):
    child = Genome()
    child.get_fitness = g1.get_fitness

    # find ending of lined up genes
    innovations_1 = [x.innovation for x in g1.connections]
    innovations_2 = [x.innovation for x in g2.connections]

    # FIXME
    debug = False
    for x in innovations_1 + innovations_2:
        if type(x) is list:
            debug = True
            break
    if debug == True:
        print('\n - new crossover error - \n')
        print('parent 1')
        print(g1, '\n')
        print('parent 2')
        print(g2, '\n')

        print('inno 1')
        print(innovations_1, '\n')
        print('inno 2')
        print(innovations_2, '\n')
        sys.exit()

    recurring = list(set(innovations_1).intersection(set(innovations_2)))

    # these lined up genes have equal chance
    for innovation in recurring:
        # flip a coin
        choice = g1 if random.uniform(0, 1) > 0.5 else g2

        # inherent gene
        child.connections.append(choice.find_connection(innovation))

    # now pull disjoint & excess genes
    if g1.fitness != g2.fitness:
        # find more fit excess genes
        if g1.fitness > g2.fitness:
            choice = g1
            innovation_choice = innovations_1
        else:
            choice = g2
            innovation_choice = innovations_2

        # inheret
        remainder = [x for x in innovation_choice if x not in recurring]
        for innovation in remainder:
            child.connections.append(choice.find_connection(innovation))
    else:
        # inherent all excess genes
        remainder_1 = [x for x in innovations_1 if x not in recurring]
        remainder_2 = [x for x in innovations_2 if x not in recurring]
        for innovation in remainder_1:
            child.connections.append(g1.find_connection(innovation))
        for innovation in remainder_2:
            child.connections.append(g2.find_connection(innovation))

    # grab neurons
    if g1.fitness != g2.fitness:
        # inheret more fit nodes
        choice = g1 if g1.fitness > g2.fitness else g2
        child.neurons = choice.neurons[:]
    else:
        # inheret all nodes
        neurons_1 = [x.id for x in g1.neurons]
        neurons_2 = [x for x in [x.id for x in g2.neurons] if x not in neurons_1]
        for neuron_id in neurons_1:
            child.neurons.append(g1.find_neuron(neuron_id))
        for neuron_id in neurons_2:
            child.neurons.append(g2.find_neuron(neuron_id))

    return child

# Mutates all weights with chance
def mutate_weights(g, chance):
    new_connections = []
    for connection in g.connections:
        # roll dice to see if a mutation is made
        if random.uniform(0,1) > (1-chance):
            # find out what type of weight change
            choice = random.uniform(0,1)

            # new number
            if choice < 0.25:
                new_connections.append(Connection(connection.input,
                                connection.output,
                                random.uniform(-1,1),
                                connection.enabled,
                                connection.innovation))

            # sign swap
            elif choice >= 0.25 and choice < 0.5:
                new_connections.append(Connection(connection.input,
                                connection.output,
                                -connection.weight,
                                connection.enabled,
                                connection.innovation))

            # percentage shift
            elif choice >= 0.5 and choice < 0.75:
                shift = random.uniform(-0.5, 0.5)
                new_connections.append(Connection(connection.input,
                                connection.output,
                                connection.weight + (connection.weight * shift),
                                connection.enabled,
                                connection.innovation))

            # activate unactive connection
            else:
                disabled_connections = [x for x in g.connections if x.enabled == False]
                if len(disabled_connections) > 0:
                    chosen = random.choice(disabled_connections)
                    new_connections.append(Connection(chosen.input,
                                    chosen.output,
                                    chosen.weight,
                                    True,
                                    chosen.innovation))

        # keep current connection
        else:
            new_connections.append(connection)

    # assign genome's new connections
    g.connections = new_connections[:]

# Adds a neuron to genome
def add_node(g):
    global Innovation
    global Innovation_Cache
    global Connection_Cache

    # grow a single neuron
    g.neurons.append(Neuron(Type.HIDDEN, g.neurons[-1].id + 1))

    # pick a connection and disable it
    choice = random.choice(g.connections)
    g.connections.remove(choice)
    g.connections.append(Connection(choice.input,
                    choice.output,
                    choice.weight,
                    False,
                    choice.innovation))

    # check if this connection has been grown before
    c_temp = [choice.input, g.neurons[-1].id]
    if c_temp in Connection_Cache:
        this_innovation = Innovation_Cache[Connection_Cache.index(c_temp)]
    else:
        Innovation += 1
        this_innovation = Innovation
        Connection_Cache.append(c_temp)
        Innovation_Cache.append(Innovation)

    # add a connection from input to new node
    g.connections.append(Connection(choice.input,
                    g.neurons[-1].id,
                    1.0,
                    True,
                    this_innovation))

    # check if this connection has been grown before
    c_temp = [g.neurons[-1].id, choice.output]
    if c_temp in Connection_Cache:
        this_innovation = Innovation_Cache[Connection_Cache.index(c_temp)]
    else:
        Innovation += 1
        this_innovation = Innovation
        Connection_Cache.append(c_temp)
        Innovation_Cache.append(Innovation)

    # add a connection from new node to output
    g.connections.append(Connection(g.neurons[-1].id,
                    choice.output,
                    choice.weight,
                    True,
                    this_innovation))

# Adds a connection to genome
def add_connection(g):
    global Innovation
    global Innovation_Cache
    global Connection_Cache

    # find already connected pairs
    connected = [[0] for x in g.neurons]
    for c in g.connections:
        connected[c.input - 1].append([c.input, c.output])
    connected = [x[1:] for x in connected]

    # flatten
    connected = [x for y in connected for x in y]

    # find unconnected pairs
    unconnected = []
    for i in range(len(g.neurons)):
        for j in range(len(g.neurons)):
            unconnected.append([g.neurons[i].id, g.neurons[j].id])
    unconnected = [x for x in unconnected if x not in connected]

    # check if this connection has been grown before
    new = random.choice(unconnected)
    if new in Connection_Cache:
        this_innovation = Innovation_Cache[Connection_Cache.index(new)]
    else:
        Innovation += 1
        this_innovation = Innovation
        Connection_Cache.append(new)
        Innovation_Cache.append(Innovation)

    # choose 1 at random
    g.connections.append(Connection(new[0],
                    new[1],
                    random.uniform(-1,1),
                    True,
                    this_innovation))

# High level dice-roller for mutations
def mutate(g, chance):
    # structural versus weight
    if random.uniform(0,1) > 0.5:
        # roll dice to see if a mutation is made
        if random.uniform(0,1) > (1-chance):
            # flip a coin to see what type it is
            if random.uniform(0,1) > 0.5:
                add_node(g)
            else:
                add_connection(g)
    else:
        mutate_weights(g, chance)

# Compatibility measure between two genomes
def compatibility(g1, g2, c1, c2, c3):
    # cache innovations
    cache1 = [x.innovation for x in g1.connections]
    cache2 = [x.innovation for x in g2.connections]

    # get length of larger genome
    l1 = len(cache1)
    l2 = len(cache2)

    N = l1 if l1 > l2 else l2
    smaller_genome = cache2 if l1 > l2 else cache1

    # common genes
    common = list(set(cache1) & set(cache2))

    # excess genes
    excess1 = [c for c in cache1 if c > max(smaller_genome)]
    excess2 = [c for c in cache2 if c > max(smaller_genome)]
    x = c1 * ( (len(excess1) + len(excess2)) / N )

    # disjoint genes
    disjoint1 = [c for c in cache1 if c not in common and c not in excess1]
    disjoint2 = [c for c in cache2 if c not in common and c not in excess2]
    y = c2 * ( (len(disjoint1) + len(disjoint2)) / N )

    # average weight difference
    W = []
    for i in range(len(common)):
        w1 = g1.find_connection(common[i]).weight
        w2 = g2.find_connection(common[i]).weight
        W.append(abs(w1 - w2))
    z = c3 * (sum(W) / len(W))

    return x + y + z



# GenePool - abstraction for evoling a collection a genomes.
class GenePool:
    def __init__(self, population_size, num_generations, cutoff, mutation, constants, logging=0):
        # Evolution Params
        self.population_size = population_size
        self.num_generations = num_generations

        # Population Cutoff
        self.cutoff = cutoff

        # Mutation Chance
        self.mutation = mutation

        # Compatibility constants
        self.c1 = constants[0]
        self.c2 = constants[1]
        self.c3 = constants[2]

        # Levels 0 -> 2
        self.logging = logging

        # Internals
        self.population = []
        self.species = []
        self.distribution = []

    def init(self, num_inputs, num_outputs, fitness_func, distribution_func):
        # Create Population
        for i in range(self.population_size):
            self.population.append(Genome().init(num_inputs, num_outputs, fitness_func))

        # Speciate TODO
        self.species = []

        # Calculate static distribution of reproduction chance
        revised_size = int(self.population_size * self.cutoff)
        distribution = [distribution_func(x) for x in range(revised_size)]
        summation = sum(distribution)
        self.distribution = [x/summation for x in distribution]

    def evolve(self):
        global Innovation_Cache, Connection_Cache
        for generation in range(self.num_generations):
            # Measure Fitness
            for genome in self.population:
                genome.evaluate()

            # Rank them
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            # logging
            if self.logging > 0:
                offset = ' ' * (len(str(self.num_generations)) - len(str(generation+1)))
                print('generation {}{} | top genome: {}'.format(generation+1, offset, self.population[0].fitness))
                if self.logging > 1:
                    print(self.population[0], '\n')

            # Eliminate Worst
            self.population = self.population[0:int(self.population_size * self.cutoff)]

            # Create offspring
            parents = np.random.choice(self.population, self.population_size*2, p=self.distribution)
            self.population = []
            i = 0
            while i < self.population_size*2:
                self.population.append(crossover(parents[i], parents[i+1]))
                mutate(self.population[-1], self.mutation)
                i += 2

            # Clear Cache
            Innovation_Cache = []
            Connection_Cache = []