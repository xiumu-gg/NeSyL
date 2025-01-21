class Node:
    def __init__(self, label):
        self.label = label
        self.children = []

    def add_child(self, child):
        self.children.append(child)

class Tableau:
    def __init__(self):
        self.nodes = []
        self.closed = False

    def add_node(self, node):
        self.nodes.append(node)

    def expand(self, knowledge_graph, type_rules):
        new_nodes = []
        for node in self.nodes:
            if node.label in knowledge_graph:
                for subject, predicate, obj in knowledge_graph[node.label]:
                    if (subject, predicate, obj) not in type_rules:
                        new_nodes.append(Node(obj))
        self.nodes.extend(new_nodes)

    def is_closed(self):
        return self.closed

    def is_exhausted(self):
        return not any(node.children for node in self.nodes)

def tableau_algorithm(knowledge_graph, type_rules, target_node):
    initial_node = Node(target_node)
    tableau = Tableau()
    tableau.add_node(initial_node)

    while not tableau.is_closed() and not tableau.is_exhausted():
        tableau.expand(knowledge_graph, type_rules)

    # 检查是否所有节点都已关闭
    for node in tableau.nodes:
        if node.label not in knowledge_graph:
            print(f"Inferred type: {node.label} is of type {node.children[0].label}")

