import os
import random
import torch
import logging
import numpy as np
import networkx as nx
import pickle

from collections import namedtuple
from torch import nn
from copy import deepcopy
from ConfigSpace.read_and_write import json as config_space_json_r_w

from naslib.search_spaces.core import primitives as ops
from naslib.utils.utils import get_project_root, AttrDict
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.darts.conversions import (
    convert_compact_to_naslib,
    convert_naslib_to_compact,
    convert_naslib_to_genotype,
    make_compact_mutable,
    make_compact_immutable,
)
from naslib.search_spaces.core.query_metrics import Metric

import torch.nn.functional as F

logger = logging.getLogger(__name__)

NUM_VERTICES = 4
NUM_OPS = 7


class DartsSearchSpace(Graph):
    """
    The search space for CIFAR-10 as defined in

        Liu et al. 2019: DARTS: Differentiable Architecture Search

    It consists of a makrograph which is predefined and not optimized
    and two kinds of learnable cells: normal and reduction cells. At
    each edge are 8 primitive operations.
    """

    """
    Scope is used to target different instances of the same cell.
    Here we divide the cells in normal/reduction cell and stage.
    This is necessary to set the correct channels at each stage.
    The architecture optimizer should consider all of them equally.
    """
    OPTIMIZER_SCOPE = [
        "activation_stage_1",
        "activation_stage_2",
        "activation_stage_3"
    ]

    QUERYABLE = True

    def __init__(self):
        """
        Initialize a new instance of the DARTS search space.
        Note:
            __init__ cannot take any parameters due to the way networkx is implemented.
            If we want to change the number of classes set a static attribute `NUM_CLASSES`
            before initializing the class. Default is 10 as for cifar-10.
        """
        super().__init__()

        activation_cell = Graph()
        activation_cell.name = (
            "activation_cell"  # Use the same name for all cells with shared attributes
        )
        # Input nodes
        activation_cell.add_node(1)
        # Intermediate nodes
        activation_cell.add_node(2)
        # Output node
        activation_cell.add_node(3)

        # Edges
        activation_cell.add_edge(1, 2)
        activation_cell.add_edges_from([(2, 3, EdgeData().finalize())])

        # set the cell name for all edges. This is necessary to convert a genotype to a naslib object
        for _, _, edge_data in activation_cell.edges.data():
            if not edge_data.is_final():
                edge_data.set("cell_name", "activation_cell")

        # Makrograph definition
        #
        self.name = "makrograph"

        self.add_node(1)  # input node
        self.add_node(2)  # preprocessing
        self.add_node(3, subgraph=activation_cell.copy().set_scope("activation_stage_1").set_input([2]))
        self.add_node(4, subgraph=activation_cell.copy().set_scope("activation_stage_2").set_input([3]))
        self.add_node(5, subgraph=activation_cell.copy().set_scope("activation_stage_3").set_input([4]))
        self.add_node(6) # output

        # chain connections
        self.add_edges_from([(i, i + 1) for i in range(1, 7)])

        # pre-processing
        self.edges[1, 2].set("op", ops.Sequential(nn.Conv2d(3, 6, 5),
                                                  nn.MaxPool2d(2),
                                                  nn.Conv2d(6, 16, 5),
                                                  nn.MaxPool2d(2),
                                                  nn.Flatten()))

        # Identy functions on all others edges
        for i in range(2, 6):
            self.edges[i, i+1].set("ops", ops.Identity())

        # set cell ops
        for i, (in_dim, out_dim) in enumerate([(16 * 5 * 5, 120), (120, 84), (84, 10)]):
            self.update_edges(
                update_func=lambda edge: _set_ops(edge, in_dim, out_dim),
                scope=f"activation_stage_{i+1}",
                private_edge_data=True,
            )


def _set_ops(edge, in_dim, out_dim):
    """
    Replace the 'op' at the edges with the ones defined here.
    This function is called by the framework for every edge in
    the defined scope.
    Args:
        current_egde_data (EdgeData): The data that currently sits
            at the edge.
    Returns:
        EdgeData: the updated EdgeData object.
    """
    if out_dim != 10:
        edge.data.set(
            "op",
            [ops.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()),
             ops.Sequential(nn.Linear(in_dim, out_dim), nn.Hardswish()),
             ops.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU()),
             ops.Sequential(nn.Linear(in_dim, out_dim), nn.Identity())
             ]
        )
    else:
        edge.data.set(
            "op",
            [ops.Sequential(nn.Linear(in_dim, out_dim), nn.Softmax())
             ]
        )
