class Node(object):
    def __init__(self, state, terminal=False):
        self.state = state
        self.value = 0
        self.children = []
        self.visits = 0
        self.terminal = terminal

def terminal(node):
    index = node.state
    for i in range(2):
        if index[i] == 0 or index[i] == N: return True
    return False

def UCB(cp):
    def select_best_child(node):
        selected   = 0
        best_value = -1e6

        for n in node.children:
            visit_ratio = np.log(node.vists)/(n.visits + 1e-7)
            v           = n.value + cp*np.sqrt(visit_ratio)

            if v > best_value:
                best_value = v
                selected = n

        return n
    return select_best_child

def get_get_children(N):
    def get_children(node):
        index = node.state

        children = []
        if not index[0] == 0:
            new_index = (index[0]-1,index[1])
            n         = Node(new_index, terminal=False)
            children.append(n)
        else
            new_index = (index[0], index[1])
            n         = Node(new_index, terminal=True)
            children.append(n)

        if not index[1] == N:
            new_index = (index[0],index[1]+1)
            n         = Node(new_index, terminal=False)
            children.append(n)
        else
            new_index = (index[0], index[1])
            n         = Node(new_index, terminal=True)
            children.append(n)

        if not index[0] == N:
            new_index = (index[0]+1,index[1])
            n         = Node(new_index, terminal=False)
            children.append(n)
        else
            new_index = (index[0], index[1])
            n         = Node(new_index, terminal=True)
            children.append(n)

        if not index[1] == 0:
            new_index = (index[0],index[1]-1)
            n         = Node(new_index, terminal=False)
            children.append(n)
        else
            new_index = (index[0], index[1])
            n         = Node(new_index, terminal=True)
            children.append(n)

        return children
    return get_children

class MCTS(object):
    def __init__(self, get_children, select_best_child, terminal, gamma=0.9):
        self.get_children      = get_children
        self.select_best_child = select_best_child
        self.terminal          = terminal

    def search(self, start_node):
        while not self.done_predicate():
            leaf_node  = self.tree_policy(start_node)

            leaf_state = leaf_node.state

            R          = self.simulate(leaf_state)

            self.update_nodes(leaf_node, R)

    def tree_policy(self, node):
        n = node
        self.visited = [n]

        while not self.terminal(n):
            if n.children == []:
                n.children = self.get_children(n)
                return n
            else:
                n = self.select_best_child(n)
                self.visited.append(n)
                
        return n

    def update_nodes(self, leaf_node, R):
