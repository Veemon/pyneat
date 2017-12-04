# standard library
import random
import sys

from collections import namedtuple
from enum import Enum
from math import exp

# third-party library
from tabulate import tabulate




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
            if node.type == Type.In:
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




# Genome - Network Blueprint
class Genome:
    def __init__(self, num_inputs, num_outputs):
        # information
        self.neurons = []
        self.connections = []

        # measurements
        self.network = 0
        self.fitness = 0

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
                                                    True, 0))

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
        pass

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

# Global innovation counter
Innovation = 0
Innovation_cache = []

# Creates offspring between two genomes
def crossover(g1, g2):
    child = Genome()

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
        child = choice.neurons
    else:
        # inheret all nodes
        neurons_1 = [x.id for x in g1.neurons]
        neurons_2 = [x for x in [x.id for x in g2.neurons] if x not in neurons_1]
        for neuron_id in neurons_1:
            child.neurons.append(g1.find_neuron(neuron_id))
        for neuron_id in neurons_2:
            child.neurons.append(g2.find_neuron(neuron_id))

    # sort
    child.neurons.sort(key=lambda x: x.id)
    child.connections.sort(key=lambda x: x.innovation)

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

            # percentage shift positive
            elif choice >= 0.5 and choice < 0.75:
                shift = random.uniform(0, 0.5)
                new_connections.append(Connection(connection.input,
                                connection.output,
                                connection.weight + (connection.weight * shift),
                                connection.enabled,
                                connection.innovation))

            # percentage shift negative
            else:
                shift = random.uniform(0, 0.5)
                new_connections.append(Connection(connection.input,
                                connection.output,
                                connection.weight - (connection.weight * shift),
                                connection.enabled,
                                connection.innovation))

        # keep current connection
        else:
            new_connections.append(connection)
    g.connections = new_connections

# Adds a neuron to genome
def add_node(g):
    g.neurons.append(Neuron(Type.HIDDEN, g.neurons[-1].id + 1))

    # pick a connection and disable it
    choice = random.choice(g.connections)
    g.connections.remove(choice)
    g.connections.append(Connection(choice.input,
                    choice.output,
                    choice.weight,
                    False,
                    choice.innovation))

    # add a connection from input to new node
    g.connections.append(Connection(choice.input,
                    g.neurons[-1].id,
                    1.0,
                    True,
                    0))

    # add a connection from new node to output
    g.connections.append(Connection(g.neurons[-1].id,
                    choice.output,
                    choice.weight,
                    True,
                    0))

# Adds a connection to genome
def add_connection(g):
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

    # choose 1 at random TODO: add innovation
    new = random.choice(unconnected)
    g.connections.append(Connection(new[0],
                    new[1],
                    random.uniform(-1,1),
                    True,
                    0))

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



# setup parent 1
g1 = Genome(3, 2)
print(g1)
