'''
-crossover-

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
from multiprocessing.pool import ThreadPool
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

class Saver(Enum):
    DISABLED = 0
    UNBIASED_GENOME = 1
    BIASED_GENOME = 2
    UNBIASED_POPULATION = 3
    BIASED_POPULATION = 4
    USER_DEFINED = 5



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
__Innovation_Counter = 0
__Connection_Cache = []
__Innovation_Cache = []

# Genome - Network Blueprint
class Genome:
    def __init__(self):
        # information
        self.neurons = []
        self.connections = []

        # measurements
        self.network = 0
        self.fitness = 0
        self.get_fitness = 0

        # logging information
        self.generation_counter = 0
        self.save_path = 0

    def init(self, num_inputs, num_outputs, fitness_function):
        global __Innovation_Counter

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
        __Innovation_Counter = num_inputs + num_outputs + 1

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
        return self

    def evaluate(self, counter):
        self.generation_counter = counter
        return self.build().get_fitness(self)

    def save(self, path):
        self.save_path = path

    def load(self, path):
        with open(path, 'r') as f:
            data = f.read()

        start_brace = -1
        end_brace = -1
        args = []

        # get neurons and connections
        for i, c in enumerate(data):
            if c == '{':
                start_brace = i + 1
            if c == '}':
                end_brace = i
            if start_brace != -1:
                if end_brace != -1:
                    args.append(data[start_brace:end_brace])
                    start_brace = -1
                    end_brace = -1

        # clean
        for i in range(len(args)):
            args[i] = args[i].replace('\t', '')
            args[i] = [x for x in args[i].split('\n') if len(x) > 0]

        # parse neurons
        for n in args[0]:
            n = n.replace('[', '').replace(']', '').split(',')
            new_neuron = Neuron( Type(int(n[0])), int(n[1]) )
            self.neurons.append(new_neuron)

        # parse connections
        for c in args[1]:
            c = c.replace('[', '').replace(']', '').split(',')
            new_connection = Connection(int(c[0]),
                                        int(c[1]),
                                        float(c[2]),
                                        bool(c[3]),
                                        int(c[4]))
            self.connections.append(new_connection)

        return self

    def _save_to_file(self, save_path="", generation=0):
        # Make sure no empties
        if self.save_path == 0:
            if save_path == "":
                print("Error: no directory specified to save to.")
                sys.exit()

        # Get all relevant information
        meta = [self.generation_counter, self.fitness]
        neurons = [[n.type.value, n.id] for n in self.neurons]
        connections = []
        for c in self.connections:
            str_c = [c.input, c.output, c.weight, int(c.enabled), c.innovation]
            connections.append(str_c) 

        # Pathing defaults
        if self.save_path == 0:
            this_path = save_path + "/gen_{}".format(generation) + ".save"
        else:
            this_path = self.save_path

        # Stringify
        with open(this_path, 'w') as f:
            print(meta, file=f)
            print('{', file=f)
            print('\t{', file=f)
            for n in neurons:
                print('\t\t{}'.format(n), file=f)
            print('\t}', file=f)
            print('\t{', file=f)
            for c in connections:
                print('\t\t{}'.format(c), file=f)
            print('\t}', file=f)
            print('}', file=f)

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
                new_connections.append(Connection(connection.input,
                                    connection.output,
                                    connection.weight,
                                    True,
                                    connection.innovation))


        # keep current connection
        else:
            new_connections.append(connection)

    # assign genome's new connections
    g.connections = new_connections[:]

# Adds a neuron to genome - 
def add_node(g):
    global __Innovation_Counter
    global __Innovation_Cache
    global __Connection_Cache

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
    if c_temp in __Connection_Cache:
        this_innovation = __Innovation_Cache[__Connection_Cache.index(c_temp)]
    else:
        __Innovation_Counter += 1
        this_innovation = __Innovation_Counter
        __Connection_Cache.append(c_temp)
        __Innovation_Cache.append(__Innovation_Counter)

    # add a connection from input to new node
    g.connections.append(Connection(choice.input,
                    g.neurons[-1].id,
                    1.0,
                    True,
                    this_innovation))

    # check if this connection has been grown before
    c_temp = [g.neurons[-1].id, choice.output]
    if c_temp in __Connection_Cache:
        this_innovation = __Innovation_Cache[__Connection_Cache.index(c_temp)]
    else:
        __Innovation_Counter += 1
        this_innovation = __Innovation_Counter
        __Connection_Cache.append(c_temp)
        __Innovation_Cache.append(__Innovation_Counter)

    # add a connection from new node to output
    g.connections.append(Connection(g.neurons[-1].id,
                    choice.output,
                    choice.weight,
                    True,
                    this_innovation))

# Adds a connection to genome
def add_connection(g):
    global __Innovation_Counter
    global __Innovation_Cache
    global __Connection_Cache

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

    # safety - if fully traversable, leave
    if len(unconnected) == 0:
        return

    # check if this connection has been grown before
    new = random.choice(unconnected)
    if new in __Connection_Cache:
        this_innovation = __Innovation_Cache[__Connection_Cache.index(new)]
    else:
        __Innovation_Counter += 1
        this_innovation = __Innovation_Counter
        __Connection_Cache.append(new)
        __Innovation_Cache.append(__Innovation_Counter)

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




# Thread Pool for Genome Evaluations
__pyneat_thread_pool = 0
def init_thread_pool(num_threads):
    global __pyneat_thread_pool
    __pyneat_thread_pool = ThreadPool(num_threads)

# Evaluation Invoker
def invoke_eval(arg):
    return arg[0].evaluate(arg[1])

# Parallel Evaluation Handler
def p_eval(population, counter):
    global __pyneat_thread_pool
    args = [[p,counter] for p in population]
    return __pyneat_thread_pool.map(invoke_eval, args)

# GenePool - abstraction for evoling a collection a genomes.
class GenePool:
    def __init__(self, population_size, num_generations, cutoff, mutation, constants,
                    logging=0, num_threads=1):
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

        # Parallel Evaluations
        self.parallel = True if num_threads > 1 else False
        if self.parallel == True:
            init_thread_pool(num_threads)

        # Internals
        self.population = []
        self.species = []
        self.distribution = []

        self.last_top = 0

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

    def evolve(self, saver=Saver.DISABLED, save_path='saves'):
        global __Innovation_Cache, __Connection_Cache

        # Log Experiment
        print(' ' * 3, '-'*35)
        print(' ' * 5, "Population Size:\t\t{}".format(self.population_size))
        print(' ' * 5, "Fitness Cutoff:\t\tTop {}%".format(int(self.cutoff*100)))
        print(' ' * 5, "Chance of Mutation:\t{}%".format(int(self.mutation*100)))
        print(' ' * 3, '-'*35,'\n')

        # Evolution
        for generation in range(self.num_generations):
            # Measure Fitness
            if self.parallel == True:
                terminations = p_eval(self.population, generation)
            else:
                terminations = [g.evaluate(generation) for g in self.population]

            # Rank them
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            # Save genome/population
            if saver == Saver.UNBIASED_GENOME:
                self.population[0]._save_to_file(save_path=save_path, generation=generation)

            elif saver == Saver.BIASED_GENOME:
                if self.population[0].fitness >= self.last_top:
                    self.last_top = self.population[0].fitness
                    self.population[0]._save_to_file(save_path=save_path, generation=generation)

            elif saver == Saver.UNBIASED_POPULATION:
                pass

            elif saver == Saver.BIASED_POPULATION:
                pass

            elif saver == Saver.USER_DEFINED:
                for g in self.population:
                    if g.save_path != 0:
                        g._save_to_file()

            # If terminate signal present, exit
            for term in terminations:
                if term == True:
                    sys.exit()

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
            while i < len(parents):
                child = crossover(parents[i], parents[i+1])
                mutate(child, self.mutation)
                self.population.append(child)
                i += 2

            # Clear Cache
            __Innovation_Cache = []
            __Connection_Cache = []
