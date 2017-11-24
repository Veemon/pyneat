import better_exceptions

node_in = 0
node_hidden = 1
node_out = 2

class Node:
    def __init__(self, args):
        self.id = args[0]
        self.type = args[1]

        self.cache = []
        self.next_node = []
        self.weights = []

    def new_out(self, output, weight):
        self.next_node.append(output)
        self.weights.append(weight)

    def hold(self, x):
        self.cache.append(x)

    def input(self, x):
        for i in range(len(self.next_node)):
            self.next_node[i].hold(x * self.weights[i])

    def forward(self):
        summation = sum(self.cache)
        for i in range(len(self.next_node)):
            self.next_node[i].hold(summation * self.weights[i])

class Connection:
    def __init__(self, args):
        self.node_in = args[0]
        self.node_out = args[1]
        self.weight = args[2]
        self.active = args[3]
        self.innovation = args[4]

class Genotype:
    def __init__(self):
        self.nodes = []
        self.connections = []

    def example_nodes(self):
        self.nodes = [0] * 5

        self.nodes[0] = Node([1, node_in])
        self.nodes[1] = Node([2, node_in])
        self.nodes[2] = Node([3, node_in])
        self.nodes[3] = Node([4, node_out])
        self.nodes[4] = Node([5, node_hidden])

    def example_connections(self):
        self.connections = [0] * 7

        self.connections[0] = Connection([1, 4,  0.7, True,  1])
        self.connections[1] = Connection([2, 4, -0.5, False, 2])
        self.connections[2] = Connection([3, 4,  0.5, True,  3])
        self.connections[3] = Connection([2, 5,  0.2, True,  4])
        self.connections[4] = Connection([5, 4,  0.4, True,  5])
        self.connections[5] = Connection([1, 5,  0.6, True,  6])
        self.connections[6] = Connection([4, 5,  0.6, True, 11])


class Phenotype:
    def __init__(self, genotype):
        self.routes = []
        self.genotype = genotype

        # get number of inputs
        for node in genotype.nodes:
            if node.type == node_in:
                self.routes.append(node)

        # set all node routes
        for connection in genotype.connections:
            for node in genotype.nodes:
                if connection.node_in == node.id and connection.active == True:
                    out_node = next(x for x in genotype.nodes if x.id == connection.node_out)
                    node.new_out(out_node, connection.weight)

    def forward(self, x):
        # initial wave
        for i in range(len(x)):
            self.routes[i].input(x[i])

        # following routes


def main():
    genome = Genotype()
    genome.example_nodes()
    genome.example_connections()

    phenotype = Phenotype(genome)
    phenotype.forward([0,0,0])

    for node in genome.nodes:
        print(node.id, end=' ')
        for out in node.next_node:
            print(out.id, end=' ')
        print()

if __name__ == '__main__':
    main()
