import logging
import networkx as nx
import matplotlib.pyplot as plt
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer
from naslib.utils import utils, setup_logger
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core import primitives as ops
from torch import nn
from IPython.display import clear_output
import torch
from naslib.search_spaces.core.primitives import AbstractPrimitive
from activation_sub_func.binary_func import Maximum, Minimum, Sub, Add, Mul, Div, SigMul
from activation_sub_func.unary_func import Power, Sin, Cos, Abs_op, Sign, Beta_mul, Beta_add, Log, Exp, Sinh, Cosh, \
    Tanh, Asinh, Acosh, Atan, Maximum0, Minimum0, Sigmoid, LogExp, Exp2, Erf, Sinc


class Stack(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return [x[0], x[1]]

    def get_embedded_ops(self):
        return None


class UnStack(AbstractPrimitive):
    def __init__(self, dim=1):
        super().__init__(locals())
        self.dim = dim

    def forward(self, x, edge_data=None):
        return x[self.dim]

    def get_embedded_ops(self):
        return None


class ActivationFuncResNet20SearchSpace(Graph):
    """
    https://www.researchgate.net/figure/ResNet-20-architecture_fig3_351046093
    """

    OPTIMIZER_SCOPE = [
        f"activation_{i}" for i in range(1, 20)
    ]

    QUERYABLE = False

    def __init__(self):
        super().__init__()

        # cell definition
        activation_cell = Graph()
        activation_cell.name = 'activation_cell'
        activation_cell.add_node(1)  # input node
        activation_cell.add_node(2)  # unary node / intermediate node
        activation_cell.add_node(3)  # unary node / intermediate node
        activation_cell.add_node(4)  # binary node / output node
        activation_cell.add_edges_from([(1, 2, EdgeData())])  # mutable intermediate edge
        activation_cell.add_edges_from([(1, 3, EdgeData())])  # mutable intermediate edge

        activation_cell.add_edges_from([(2, 4, EdgeData().finalize())])  # mutable intermediate edge
        activation_cell.add_edges_from([(3, 4, EdgeData().finalize())])  # mutable intermediate edge
        activation_cell.nodes[4]['comb_op'] = Stack()

        activation_cell.add_node(5)  # binary node / output node
        activation_cell.add_edges_from([(4, 5, EdgeData())])  # mutable intermediate edge

        activation_cell.add_node(6)
        activation_cell.add_edges_from([(5, 6, EdgeData().finalize())])  # unary node / intermediate node
        activation_cell.add_node(7)
        activation_cell.add_edges_from([(6, 7, EdgeData())])  # mutable intermediate edge
        activation_cell.add_node(8)
        activation_cell.add_edges_from([(1, 8, EdgeData())])  # mutable intermediate edge

        activation_cell.add_node(9)
        activation_cell.add_edges_from([(8, 9, EdgeData().finalize())])  # mutable intermediate edge
        activation_cell.add_edges_from([(7, 9, EdgeData().finalize())])  # mutable intermediate edge
        activation_cell.nodes[9]['comb_op'] = Stack()

        activation_cell.add_node(10)
        activation_cell.add_edges_from([(9, 10, EdgeData())])  # mutable intermediate edge

        activation_cell.add_node(11)
        activation_cell.add_edges_from([(10, 11, EdgeData().finalize())])  # mutable intermediate edge

        for tup in [(1, 2), (1, 3), (1, 8), (6, 7)]:  # unary operations
            activation_cell.edges[tup[0], tup[1]].set("op", [
                ops.Sequential(nn.Identity()),
                # ops.Zero(stride=1),
                # ops.Sequential(Power(2)),
                # ops.Sequential(Power(3)),
                # ops.Sequential(Power(.5)),
                ops.Sequential(Sin()),
                ops.Sequential(Cos()),
                # ops.Sequential(Abs_op()),
                ops.Sequential(Sign()),
                # Todo add channel options
                # ops.Sequential(Beta_mul(channels=32)),
                # ops.Sequential(Beta_add(channels=32)),
                # ops.Sequential(Log()),
                # ops.Sequential(Exp()),
                # ops.Sequential(Sinh()),
                # ops.Sequential(Cosh()),
                # ops.Sequential(Tanh()),
                # ops.Sequential(Asinh()),
                # ops.Sequential(Acosh()),
                # ops.Sequential(Atan()),
                # ops.Sequential(Sinc()),
                ops.Sequential(Maximum0()),
                ops.Sequential(Minimum0()),
                ops.Sequential(Sigmoid()),
                # ops.Sequential(LogExp()),
                # ops.Sequential(Exp2()),
                # ops.Sequential(Erf()),
                # Todo add channel options
                # ops.Sequential(Beta(channels=16)),
            ])

        for tup in [(4, 5), (9, 10)]:
            activation_cell.edges[tup[0], tup[1]].set("op", [
                ops.Sequential(Add()),
                ops.Sequential(Sub()),
                # ops.Sequential(Mul()),
                # ops.Sequential(Div()),
                # ops.Sequential(Maximum()),
                # ops.Sequential(Minimum()),
                # ops.Sequential(SigMul()),
                # Todo add channel options
                #                 ops.Sequential(ExpBetaSub2(channels=32)),
                #                 ops.Sequential(ExpBetaSubAbs(channels=32)),
                #                 ops.Sequential(BetaMix(channels=32)),
            ])

        # macroarchitecture definition
        self.name = 'makrograph'
        self.add_node(1)  # input
        self.add_node(2)  # intermediate
        self.add_node(3,
                      subgraph=activation_cell.copy().set_scope("activation_1").set_input([2]))  # activation cell 3
        self.nodes[3]['subgraph'].name = "activation_1"

        self.add_node(4)
        self.add_node(5,
                      subgraph=activation_cell.copy().set_scope("activation_2").set_input([4]))  # activation cell 3
        self.nodes[5]['subgraph'].name = "activation_2"

        self.add_node(6)
        # Todo add option here with a func which has a arg channels
        self.add_node(7,
                      subgraph=activation_cell.copy().set_scope("activation_3").set_input([6]))  # activation cell 3
        self.nodes[7]['subgraph'].name = "activation_3"

        self.add_edges_from([
            (1, 2, EdgeData()),
            (2, 3, EdgeData()),
            (3, 4, EdgeData()),
            (4, 5, EdgeData()),
            (5, 6, EdgeData()),
            (3, 6, EdgeData()),
            (6, 7, EdgeData())
        ])

        self.edges[1, 2].set('op',
                             ops.Sequential(nn.Conv2d(3, 16, 3, padding=1), ))  # convolutional edge
        self.edges[3, 4].set('op',
                             ops.Sequential(nn.Conv2d(16, 16, 3, padding=1), ))  # convolutional edge
        self.edges[5, 6].set('op',
                             ops.Sequential(nn.Conv2d(16, 16, 3, padding=1), ))  # convolutional edge

        conv_option = {
            "in_channels": 16,
            "out_channels": 16,
            "kernel_size": 3,
            "padding": 1
        }
        self._create_base_block(7, 4, activation_cell, conv_option)
        self._create_base_block(11, 6, activation_cell, conv_option)

        conv_option_a = {
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 3,
            "padding": 1,
            "stride": 2
        }
        conv_option_b = {
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 1,
            "padding": 0,
            "stride": 2
        }
        self._create_reduction_block(15, 8, activation_cell, conv_option_a, conv_option_b)

        conv_option = {
            "in_channels": 32,
            "out_channels": 32,
            "kernel_size": 3,
            "padding": 1
        }
        self._create_base_block(19, 10, activation_cell, conv_option)
        self._create_base_block(23, 12, activation_cell, conv_option)

        conv_option_a = {
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "stride": 2
        }
        conv_option_b = {
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 1,
            "padding": 0,
            "stride": 2
        }
        self._create_reduction_block(27, 14, activation_cell, conv_option_a, conv_option_b)

        conv_option = {
            "in_channels": 64,
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1
        }
        self._create_base_block(31, 16, activation_cell, conv_option)
        self._create_base_block(34, 18, activation_cell, conv_option)

        # add head
        self.add_node(39)
        self.add_edges_from([
            (38, 39, EdgeData())
        ])
        self.edges[38, 39].set('op',
                               ops.Sequential(
                                   nn.AvgPool2d(8),
                                   nn.Flatten(),
                                   nn.Linear(64, 10),
                                   nn.Softmax()
                               ))  # convolutional edge
        self.add_node(40)
        self.add_edges_from([
            (39, 40, EdgeData().finalize())
        ])

    def _create_base_block(self, start: int, stage: int, cell, conv_option: dict):
        self.add_node(start + 1)

        self.add_node(start + 2, subgraph=cell.copy().set_scope(f"activation_{stage}").set_input(
            [start + 1]))  # activation cell 3
        self.nodes[start + 2]['subgraph'].name = f"activation_{stage}"

        self.add_node(start + 3)

        self.add_node(start + 4, subgraph=cell.copy().set_scope(f"activation_{stage + 1}").set_input(
            [start + 3]))  # activation cell 3
        self.nodes[start + 4]['subgraph'].name = f"activation_{stage + 1}"

        self.add_edges_from([
            (start, start + 1, EdgeData()),
            (start, start + 3, EdgeData()),
            (start + 1, start + 2, EdgeData()),
            (start + 2, start + 3, EdgeData()),
            (start + 3, start + 4, EdgeData()),
        ])

        self.edges[start, start + 1].set('op',
                                         ops.Sequential(nn.Conv2d(**conv_option), ))  # convolutional edge
        self.edges[start + 2, start + 3].set('op',
                                             ops.Sequential(nn.Conv2d(**conv_option), ))  # convolutional edge

    def _create_reduction_block(self, start: int, stage: int, cell, conv_option_a: dict, conv_option_b: dict):
        self.add_node(start + 1)

        self.add_node(start + 2, subgraph=cell.copy().set_scope(f"activation_{stage}").set_input(
            [start + 1]))  # activation cell 3
        self.nodes[start + 2]['subgraph'].name = f"activation_{stage}"

        self.add_node(start + 3)

        self.add_node(start + 4, subgraph=cell.copy().set_scope(f"activation_{stage + 1}").set_input(
            [start + 3]))  # activation cell 3
        self.nodes[start + 4]['subgraph'].name = f"activation_{stage + 1}"

        self.add_edges_from([
            (start, start + 1, EdgeData()),
            (start, start + 3, EdgeData()),  # add conv
            (start + 1, start + 2, EdgeData()),
            (start + 2, start + 3, EdgeData()),
            (start + 3, start + 4, EdgeData()),
        ])

        self.edges[start, start + 1].set('op',
                                         ops.Sequential(nn.Conv2d(**conv_option_a), ))  # convolutional edge
        conv_option_a["in_channels"] = conv_option_a["out_channels"]
        conv_option_a["stride"] = 1

        self.edges[start, start + 3].set('op',
                                         ops.Sequential(nn.Conv2d(**conv_option_b), ))  # convolutional edge
        self.edges[start + 2, start + 3].set('op',
                                             ops.Sequential(nn.Conv2d(**conv_option_a), ))  # convolutional edge

    def _set_ops(self, edge):
        edge.data.set('op', [
            ops.Sequential(nn.ReLU()),
            ops.Sequential(nn.Hardswish()),
            ops.Sequential(nn.LeakyReLU()),
            ops.Sequential(nn.Identity())
        ])


config = utils.get_config_from_args(config_type='nas')
config.optimizer = 'darts'
config.search.batch_size = 32
config.search.learning_rate = 0.01
utils.set_seed(config.seed)
clear_output(wait=True)
utils.log_args(config)

logger = setup_logger(config.save + '/log.log')
logger.setLevel(logging.INFO)

search_space = ActivationFuncResNet20SearchSpace()
# nx.draw_kamada_kawai(search_space)
# plt.show()

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search()

trainer.evaluate_oneshot()
