import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer
from naslib.search_spaces import DartsSearchSpace
from naslib.utils import utils, setup_logger, get_config_from_args, set_seed, log_args
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core import primitives as ops
from torch import nn
from fvcore.common.config import CfgNode
from copy import deepcopy
from IPython.display import clear_output
import torch
from naslib.search_spaces.core.primitives import AbstractPrimitive




class Power(AbstractPrimitive):
    def __init__(self, power):
        super().__init__(locals())
        self.power = power

    def forward(self, x, edge_data=None):
        return torch.pow(x, self.power)

    def get_embedded_ops(self):
        return None


class Sin(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.sin(x)

    def get_embedded_ops(self):
        return None


class Cos(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.cos(x)

    def get_embedded_ops(self):
        return None


class Abs_op(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.abs(x)

    def get_embedded_ops(self):
        return None


class Sign(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return x * -1

    def get_embedded_ops(self):
        return None


class Beta_mul(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        return x * self.beta

    def get_embedded_ops(self):
        return None


class Beta_add(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        return x + self.beta

    def get_embedded_ops(self):
        return None


class Log(AbstractPrimitive):
    def __init__(self, eps=1e-10):
        super().__init__(locals())
        self.eps = eps

    def forward(self, x, edge_data=None):
        return torch.log(x + self.eps)

    def get_embedded_ops(self):
        return None


class Exp(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.exp(x)

    def get_embedded_ops(self):
        return None


class Sinh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.sinh(x)

    def get_embedded_ops(self):
        return None


class Cosh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.cosh(x)

    def get_embedded_ops(self):
        return None


class Tanh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.tanh(x)

    def get_embedded_ops(self):
        return None


class Asinh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.asinh(x)

    def get_embedded_ops(self):
        return None


class Acosh(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.acosh(x)

    def get_embedded_ops(self):
        return None


class Atan(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.atan(x)

    def get_embedded_ops(self):
        return None


class Sinc(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.sinc(x)

    def get_embedded_ops(self):
        return None


class Maximum0(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.maximum(x, torch.zeros(x.shape).cuda())

    def get_embedded_ops(self):
        return None


class Minimum0(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.minimum(x, torch.zeros(x.shape).cuda())

    def get_embedded_ops(self):
        return None


class Sigmoid(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.sigmoid(x)

    def get_embedded_ops(self):
        return None


class LogExp(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.log(1 + torch.exp(x))

    def get_embedded_ops(self):
        return None


class Exp2(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.exp(-torch.pow(x, 2))

    def get_embedded_ops(self):
        return None


class Erf(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.erf(x)

    def get_embedded_ops(self):
        return None


class Beta(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        return self.beta

    def get_embedded_ops(self):
        return None


class Add(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.add(x[0], x[1])

    def get_embedded_ops(self):
        return None


class Sub(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.sub(x[0], x[1])

    def get_embedded_ops(self):
        return None


class Mul(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.mul(x[0], x[1])

    def get_embedded_ops(self):
        return None


class Div(AbstractPrimitive):
    def __init__(self, eps=1e-10):
        super().__init__(locals())
        self.eps = eps

    def forward(self, x, edge_data=None):
        return torch.div(x[0], x[1] + self.eps)

    def get_embedded_ops(self):
        return None


class Maximum(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.maximum(x[0], x[1])

    def get_embedded_ops(self):
        return None


class Minimum(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.minimum(x[0], x[1])

    def get_embedded_ops(self):
        return None


class SigMul(AbstractPrimitive):
    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.mul(torch.sigmoid(x[0]), x[1])

    def get_embedded_ops(self):
        return None


class ExpBetaSub2(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        return torch.exp(-self.beta * torch.pow(torch.sub(x[0], x[1]), 2))

    def get_embedded_ops(self):
        return None


class ExpBetaSubAbs(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        return torch.exp(-self.beta * torch.abs(torch.sub(x[0], x[1])))

    def get_embedded_ops(self):
        return None


class BetaMix(AbstractPrimitive):
    def __init__(self, channels):
        super().__init__(locals())
        self.beta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x, edge_data=None):
        return torch.add(-self.beta * x[0], (1 - self.beta) * x[1])

    def get_embedded_ops(self):
        return None


class stack():
    def __init__(self):
        pass
    def __call__(self, tensors, edges_data=None):
        return torch.stack(tensors)


class RNNResNet20SearchSpace(Graph):
    """
    https://www.researchgate.net/figure/ResNet-20-architecture_fig3_351046093
    """

    OPTIMIZER_SCOPE = [
        f"activation_{i}" for i in range(1, 20)
    ]

    OPTIMIZER_SCOPE += [
        "u_stage_1",
        "u_stage_2",
        "u_stage_3",
        "u_stage_4",
        "b_stage_1",
        "b_stage_2"
    ]

    QUERYABLE = False

    def __init__(self):
        super().__init__()

        # unary cell definition
        unary_cell = Graph()
        unary_cell.name = 'u_cell'
        unary_cell.add_node(1)  # input node
        unary_cell.add_node(2)  # intermediate node
        unary_cell.add_node(3)  # output node
        unary_cell.add_edges_from([(1, 2, EdgeData())])  # mutable edge
        unary_cell.edges[1, 2].set('cell_name', 'u_cell')
        unary_cell.add_edges_from([(2, 3, EdgeData().finalize())])  # immutable edge

        # binary cell definition
        binary_cell = Graph()
        binary_cell.name = 'b_cell'
        binary_cell.add_node(1)  # input node
        binary_cell.add_node(2)  # input node
        binary_cell.add_node(3)  # concatination node
        binary_cell.nodes[3]['comb_op'] = stack()
        binary_cell.add_node(4)  # intermediate node
        binary_cell.add_node(5)  # output node
        binary_cell.add_edges_from([(3, 4, EdgeData())])  # mutable edge
        binary_cell.edges[3, 4].set('cell_name', 'b_cell')
        binary_cell.add_edges_from([(1, 3, EdgeData().finalize()),
                                    (2, 3, EdgeData().finalize()),
                                    (4, 5, EdgeData().finalize())])  # immutable edges

        # activation cell definition
        activation_cell = Graph()
        activation_cell.name = 'a_cell'
        activation_cell.add_node(1)  # input node

        activation_cell.add_node(2, subgraph=deepcopy(unary_cell).set_scope('u_stage_1').set_input([1]))  # unary cell 1
        activation_cell.nodes[2]['subgraph'].name = 'u_stage_1'
        # u_stage
        activation_cell.add_node(2)  # input node
        activation_cell.add_node(3)  # intermediate node
        activation_cell.add_node(4)  # output node
        activation_cell.add_edges_from([(1, 2, EdgeData())])  # mutable edge
        activation_cell.add_edges_from([(2, 3, EdgeData())])  # mutable edge
        activation_cell.add_edges_from([(3, 4, EdgeData())])  # immutable edge

        activation_cell.add_node(3, subgraph=deepcopy(unary_cell).set_scope('u_stage_2').set_input([1]))  # unary cell 2
        activation_cell.nodes[3]['subgraph'].name = 'u_stage_2'
        # u_stage
        activation_cell.add_node(5)  # input node
        activation_cell.add_node(6)  # intermediate node
        activation_cell.add_node(7)  # output node
        activation_cell.add_edges_from([(1, 5, EdgeData())])  # mutable edge
        activation_cell.add_edges_from([(5, 6, EdgeData())])  # mutable edge
        activation_cell.add_edges_from([(6, 7, EdgeData())])  # immutable edge

        activation_cell.add_node(4, subgraph=deepcopy(unary_cell).set_scope('u_stage_3').set_input([1]))  # unary cell 3
        activation_cell.nodes[4]['subgraph'].name = 'u_stage_3'
        # u_stage
        activation_cell.add_node(8)  # input node
        activation_cell.add_node(9)  # intermediate node
        activation_cell.add_node(10)  # output node
        activation_cell.add_edges_from([(1, 8, EdgeData())])  # mutable edge
        activation_cell.add_edges_from([(8, 9, EdgeData())])  # mutable edge
        activation_cell.add_edges_from([(9, 10, EdgeData())])  # immutable edge

        activation_cell.add_node(5, subgraph=deepcopy(binary_cell).set_scope('b_stage_1').set_input(
            [2, 3]))  # binary cell 1, 1,4
        activation_cell.nodes[5]['subgraph'].name = 'b_stage_1'
        # b_cell
        binary_cell.add_node(11)  # input node
        binary_cell.add_node(12)  # input node
        binary_cell.add_node(13)  # concatination node
        binary_cell.nodes[13]['comb_op'] = stack()
        binary_cell.add_node(14)  # intermediate node
        binary_cell.add_node(15)  # output node
        binary_cell.add_edges_from([(4, 11, EdgeData()),  # input
                                    (7, 12, EdgeData()),  # input
                                    (11, 13, EdgeData()),
                                    (12, 13, EdgeData()),
                                    (13, 14, EdgeData()),
                                    (14, 15, EdgeData())])  # immutable edges


        activation_cell.add_node(6, subgraph=deepcopy(unary_cell).set_scope('u_stage_4').set_input([5]))  # unary cell 4
        activation_cell.nodes[6]['subgraph'].name = 'u_stage_4'
        # u_stage
        activation_cell.add_node(16)  # input node
        activation_cell.add_node(17)  # intermediate node
        activation_cell.add_node(18)  # output node
        activation_cell.add_edges_from([(15, 16, EdgeData())])  # mutable edge
        activation_cell.add_edges_from([(16, 17, EdgeData())])  # mutable edge
        activation_cell.add_edges_from([(17, 18, EdgeData())])  # immutable edge

        activation_cell.add_node(7, subgraph=deepcopy(binary_cell).set_scope('b_stage_2').set_input(
            [10, 18]))  # binary cell 2
        # b_cell
        binary_cell.add_node(19)  # input node
        binary_cell.add_node(20)  # input node
        binary_cell.add_node(21)  # concatination node
        binary_cell.nodes[21]['comb_op'] = stack()
        binary_cell.add_node(22)  # intermediate node
        binary_cell.add_node(23)  # output node
        binary_cell.add_edges_from([(10, 19, EdgeData()),  # input
                                    (18, 20, EdgeData()),  # input
                                    (19, 21, EdgeData()),
                                    (20, 21, EdgeData()),
                                    (21, 22, EdgeData()),
                                    (22, 23, EdgeData())])  # immutable edges


        activation_cell.nodes[7]['subgraph'].name = 'b_stage_2'

        activation_cell.add_node(24)  # output node
        activation_cell.add_edges_from([(23, 24, EdgeData().finalize())])

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

        for scope in range(1, 4):
            self.update_edges(
                update_func=lambda edge: self._set_ops(edge),
                scope=f"activation_{scope}",
                private_edge_data=True,
            )

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

        self.update_edges(
            update_func=lambda edge: self._set_ops(edge),
            scope=f"activation_{stage}",
            private_edge_data=True, )

        self.update_edges(
            update_func=lambda edge: self._set_ops(edge),
            scope=f"activation_{stage + 1}",
            private_edge_data=True, )

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

        self.update_edges(
            update_func=lambda edge: self._set_ops(edge),
            scope=f"activation_{stage}",
            private_edge_data=True, )

        self.update_edges(
            update_func=lambda edge: self._set_ops(edge),
            scope=f"activation_{stage + 1}",
            private_edge_data=True, )

    def _set_ops(self, edge):
        if edge.data['cell_name'] == 'u_cell':
            edge.data.set('op', [
                ops.Identity(),
                ops.Zero(stride=1)
            ])
        elif edge.data['cell_name'] == 'b_cell':
            edge.data.set('op', [
                Minimum(),
                Maximum()
            ])


config = utils.get_config_from_args(config_type='nas')
config.optimizer = 'darts'
utils.set_seed(config.seed)
clear_output(wait=True)
utils.log_args(config)

logger = setup_logger(config.save + '/log.log')
logger.setLevel(logging.INFO)

search_space = RNNResNet20SearchSpace()

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search()

trainer.evaluate_oneshot()
