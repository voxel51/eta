"""
Core infrastructure for graph manipulation.

Copyright 2017-2022, Voxel51, Inc.
voxel51.com
"""
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import copy


class DirectedGraph(object):
    """A simple, lightweight implementation of a directed graph.

    The graph nodes can be any hashable objects (they are stored internally as
    keys of dictionaries and elements of sets)
    """

    def __init__(self):
        self._sources = {}
        self._sinks = {}

    @property
    def is_empty(self):
        """Returns True/False if the graph is empty."""
        return len(self._sinks) == 0

    def add_edge(self, source, sink):
        """Adds a directed edge from `source` to `sink`."""
        _add(self._sources, sink, source)
        _add(self._sinks, source, sink)

    def remove_edge(self, source, sink):
        """Removes the directed edge from `source` to `sink`."""
        _remove(self._sources, sink, source)
        _remove(self._sinks, source, sink)

    def get_graph_sources(self):
        """Returns the set of graph source nodes, i.e., nodes with no incoming
        edges.
        """
        return set(self._sinks.keys()) - set(self._sources.keys())

    def get_sources(self, node):
        """Returns the set of source nodes for the given node."""
        try:
            return self._sources[node]
        except KeyError:
            return set()

    def get_sinks(self, node):
        """Returns the set of sink nodes for the given node."""
        try:
            return self._sinks[node]
        except KeyError:
            return set()

    def sort(self):
        """Performs a topological sort of the graph nodes.

        Returns:
            a list defining a (not necessarily unique) ordering of the nodes
            such that if source -> sink, then source appears before sink in
            the list

        Raises:
            CyclicGraphError: if the graph contains a cycle
        """
        return kahns_algorithm(copy.deepcopy(self))


def kahns_algorithm(graph):
    """Runs Kahn's algorithm on the DirectedGraph.

    The input graph is destroyed, so you should pass in a deep copy of your
    graph.

    Args:
        graph: a DirectedGraph instance

    Returns:
        a list containing a (not necessarily unique) topological ordering of
        the nodes in the graph

    Raises:
        CyclicGraphError: if the graph contains a cycle
    """
    order = []
    heads = graph.get_graph_sources()
    while heads:
        node = heads.pop()
        order.append(node)
        for sink in graph.get_sinks(node).copy():
            graph.remove_edge(node, sink)
            if not graph.get_sources(sink):
                heads.add(sink)
    if not graph.is_empty:
        raise CyclicGraphError("Graph contains a cycle.")
    return order


def _add(d, key, val):
    if key not in d:
        d[key] = set()
    d[key].add(val)


def _remove(d, key, val):
    d[key].remove(val)
    if not d[key]:
        del d[key]


class CyclicGraphError(Exception):
    pass
