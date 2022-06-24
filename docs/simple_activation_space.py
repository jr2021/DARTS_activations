from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core import primitives as ops
from torch import nn
from copy import deepcopy

class DartsSearchSpace(Graph):

    OPTIMIZER_SCOPE = [
        'a_stage_1',
        'a_stage_2', 
        'a_stage_3'
    ]

    QUERYABLE = False

    def __init__(self):
        super().__init__()

        channels = [(16 * 5 * 5, 120), (120, 84), (84, 10)]
        stages = ['a_stage_1', 'a_stage_2', 'a_stage_3']

        # cell definition
        activation_cell = Graph()
        activation_cell.name = 'activation_cell'
        activation_cell.add_node(1) # input node
        activation_cell.add_node(2) # intermediate node
        activation_cell.add_node(3) # output node

        activation_cell.add_edges_from([(1, 2, EdgeData())]) # mutable intermediate edge
        activation_cell.edges[1, 2].set('cell_name', 'activation_cell') 
        activation_cell.add_edges_from([(2, 3, EdgeData().finalize())]) # immutable output edge

        # macroarchitecture definition
        self.name = 'makrograph'
        self.add_node(1) # input node
        self.add_node(2) # intermediate node
        self.add_node(3, subgraph=deepcopy(activation_cell).set_scope('a_stage_1').set_input([2])) # activation node 1
        self.add_node(4, subgraph=deepcopy(activation_cell).set_scope('a_stage_2').set_input([3])) # activation node 2
        self.add_node(5, subgraph=deepcopy(activation_cell).set_scope('a_stage_3').set_input([4])) # activation node 3
        self.nodes[3]['subgraph'].name = self.nodes[3]['subgraph'].scope
        self.nodes[4]['subgraph'].name = self.nodes[4]['subgraph'].scope
        self.nodes[5]['subgraph'].name = self.nodes[5]['subgraph'].scope
        self.add_node(6) # output node

        self.add_edges_from([(i, i+1, EdgeData()) for i in range(1, 6)])
        self.edges[1, 2].set('op',
            ops.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.MaxPool2d(2),
                nn.Flatten()
            )) # convolutional edge
        
        for stage, (in_dim, out_dim) in zip(stages, channels):
            self.update_edges(
                update_func=lambda edge: self._set_ops(edge, in_dim, out_dim),
                scope=stage,
                private_edge_data=True,
            )

    def _set_ops(self, edge, in_dim, out_dim):
        if out_dim != 10:
            edge.data.set('op', [
                ops.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()),
                ops.Sequential(nn.Linear(in_dim, out_dim), nn.Hardswish()),
                ops.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU()),
                ops.Sequential(nn.Linear(in_dim, out_dim), nn.Identity())
            ])
        else:
            edge.data.set('op', [
                ops.Sequential(nn.Linear(in_dim, out_dim), nn.Softmax(dim=1))
            ])  

      