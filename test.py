import better_exceptions

type_input = 0
hidden = 1
output = 2

class Node:
    def __init__(self, args):
        self.id = args[0]
        self.type = args[1]

class Connection:
    def __init__(self, args):
        self.node_in = args[0]
        self.node_out = args[1]

class Genotype:
    def __init__(self):
        self.nodes = []
        self.connections = []

    def example_nodes(self):
        self.nodes = [0] * 5

        self.nodes[0] = Node([1, type_input])
        self.nodes[1] = Node([2, type_input])
        self.nodes[2] = Node([3, type_input])
        self.nodes[3] = Node([4, output])
        self.nodes[4] = Node([5, hidden])

    def example_connections(self):
        self.connections = [0] * 6

        self.connections[0] = Connection([1, 4])
        self.connections[1] = Connection([3, 4])
        self.connections[2] = Connection([2, 5])
        self.connections[3] = Connection([5, 4])
        self.connections[4] = Connection([1, 5])
        self.connections[5] = Connection([4, 5])

def test(input_list):
    for i in range(1, len(input_list)):
        for j in range(len(input_list[0])):
            if input_list[i][-1] == input_list[0][j]:
                input_list[0].insert(j, input_list[i][:j])
                input_list[i][:j] = input_list[i][j+1:]

                print(input_list[i][:j])
                input()
    return input_list

class Phenotype:
    def __init__(self, genotype):
        self.routes = []
        for con in genotype.connections:
            self.routes.append([con.node_in, con.node_out])

        print(self.routes)
        print(test(self.routes))

def main():
    genome = Genotype()
    genome.example_nodes()
    genome.example_connections()

    phenotype = Phenotype(genome)

if __name__ == '__main__':
    main()
