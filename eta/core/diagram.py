'''
Core utilities for building block diagrams of modules and pipelines

Copyright 2017-2018, Voxel51, LLC
voxel51.com

Brian Moore, brian@voxel51.com
'''
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *
from future.utils import iteritems, itervalues
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import os

import eta.core.utils as etau


logger = logging.getLogger(__name__)


BLOCKDIAG_EXECUTABLE = "blockdiag"
BLOCKDIAG_INDENTATION = 2
PARAM_COLOR = "\"#EE7531\""  # Voxel51 orange
MODULE_COLOR = "\"#AAAAAA\""  # Voxel51 gray
MODULE_HEIGHT = 60
NODE_HEIGHT = 40


class BlockDiagram(object):
    '''Mixin class for objects that can be rendered as block diagrams.'''

    def _to_blockdiag(self, path):
        raise NotImplementedError("subclass must implement _to_blockdiag()")

    def render(self, path):
        '''Render the block diagram as an SVG image.

        Args:
            path: the output path, which should have extension ".svg"
        '''
        # Parse path
        base, ext = os.path.splitext(path)
        blockdiag_path = base + ".diag"
        svg_path = base + ".svg"
        if ext != ".svg":
            logger.warning("Replacing extension of '%s' with '.svg'", path)

        # Generate block diagram
        logger.info("Generating blockdiag file '%s'", blockdiag_path)
        self._to_blockdiag(blockdiag_path)

        # Render SVG
        logger.info("Rendering block diagram image '%s'", svg_path)
        args = [BLOCKDIAG_EXECUTABLE, "-Tsvg", "-o", svg_path, blockdiag_path]
        etau.communicate_or_die(args)


class BlockdiagFile(object):
    '''Base class for representing a block diagram in the blockdiag syntax.'''

    def __init__(self):
        '''Creates a BlockdiagFile instance.'''
        self._file = BlockdiagGroup(is_top_level=True)
        self._modules = BlockdiagSection(comment="modules")
        self._inputs = BlockdiagSection(comment="inputs")
        self._outputs = BlockdiagSection(comment="outputs")
        self.parameters = BlockdiagSection(comment="parameters")
        self._io_edges = BlockdiagSection(comment="I/O connections")
        self._param_edges = BlockdiagSection(comment="parameter connections")
        self._param_groups = {}

        self._file.add_element(self._modules)
        self._file.add_element(self._inputs)
        self._file.add_element(self._outputs)
        self._file.add_element(self.parameters)
        self._file.add_element(self._io_edges)
        self._file.add_element(self._param_edges)

    def write(self, path):
        '''Writes the block diagram to disk, creating the output directory if
        necessary.
        '''
        etau.ensure_basedir(path)
        with open(path, "wb") as f:
            f.write(self._file.render())

    def _add_module(self, name, shape="box"):
        node = BlockdiagNode(
            name, shape=shape,
            height=MODULE_HEIGHT, width=_module_extent(name))
        self._modules.add_element(node)

        group = BlockdiagGroup(orientation="portrait", color=PARAM_COLOR)
        self._param_edges.add_element(group)
        self._param_groups[name] = group

    def _add_input(self, name, shape="endpoint"):
        node = BlockdiagNode(
            name, shape=shape, height=NODE_HEIGHT, width=_node_extent(name))
        self._inputs.add_element(node)

    def _add_output(self, name, shape="endpoint"):
        node = BlockdiagNode(
            name, shape=shape, height=NODE_HEIGHT, width=_node_extent(name))
        self._outputs.add_element(node)

    def _add_parameter(self, name, shape="beginpoint"):
        node = BlockdiagNode(
            name, shape=shape, rotate=270, height=_node_extent(name),
            width=NODE_HEIGHT)
        self.parameters.add_element(node)

    def _add_io_edge(self, source, sink):
        self._io_edges.add_element(BlockdiagDirectedEdge(source, sink))

    def _add_param_edge(self, param_name, module_name):
        self._param_groups[module_name].add_element(
            BlockdiagDirectedEdge(param_name, module_name))


class BlockdiagModule(BlockdiagFile):
    '''Class representing a blockdiag module.'''

    def __init__(self, name):
        '''Creates a new block diagram module.

        Args:
            name: the name of the module
        '''
        super(BlockdiagModule, self).__init__()
        self.name = name
        self._add_module(name)

    def add_input(self, name):
        '''Add a module input with the given name.'''
        self._add_input(name)
        self._add_io_edge(name, self.name)

    def add_output(self, name):
        '''Add a module output with the given name.'''
        self._add_output(name)
        self._add_io_edge(self.name, name)

    def add_parameter(self, name):
        '''Add a module parameter with the given name.'''
        self._add_parameter(name)
        self._add_param_edge(name, self.name)


class BlockdiagElement(object):
    '''Interface for all elements of a block diagram file.'''

    def render(self, indent=0):
        '''Renders the element as a string at the given indentiation level.

        Args:
            indent: the indentation level at which to render the element
        '''
        raise NotImplementedError("subclass must implement render()")


class BlockdiagContainer(BlockdiagElement):
    '''Base class for block digram elements that contain other elements.'''

    def __init__(self):
        '''Creates a new empty block diagram container.'''
        self.elements = []

    @property
    def is_empty(self):
        '''Returns True/False if the container is empty.'''
        return len(self.elements) == 0

    def add_element(self, element):
        '''Adds the element to the container.'''
        self.elements.append(element)


class BlockdiagGroup(BlockdiagContainer):
    '''A group of elements.'''

    def __init__(self, is_top_level=False, **kwargs):
        '''Creates a new block diagram group.

        Args:
            is_top_level: whether this is the top-level blockdiag group or a
                nested group
            **kwargs: an optional dictionary of key = value attributes for
                the group
        '''
        super(BlockdiagGroup, self).__init__()
        self.name = "blockdiag" if is_top_level else "group"
        self.attributes = [
            BlockdiagAttribute(k, v) for k, v in iteritems(kwargs)
        ]

    def render(self, indent=0):
        if self.is_empty:
            return ""
        c = [_indent(self.name + " {", indent)]
        for a in self.attributes:
            c.append(a.render(indent=indent + BLOCKDIAG_INDENTATION))
        for e in self.elements:
            c.append(e.render(indent=indent + BLOCKDIAG_INDENTATION))
        c.append(_indent("}", indent))
        return "\n".join(c).rstrip()


class BlockdiagSection(BlockdiagContainer):
    '''A list of elements with an optional comment above.'''

    def __init__(self, comment=None):
        '''Creates a new block diagram section.

        Args:
            comment: an optional comment to place above the section
        '''
        super(BlockdiagSection, self).__init__()
        if comment:
            self.add_element(BlockdiagComment(comment))

    def render(self, indent=0):
        c = []
        for e in self.elements:
            c.append(e.render(indent=indent))
        content = "\n".join(c).rstrip()
        return ("\n" if content else "") + content


class BlockdiagComment(BlockdiagElement):
    '''A comment element.'''

    def __init__(self, comment):
        '''Creates a new block diagram comment.

        Args:
            comment: a string
        '''
        self.comment = comment

    def render(self, indent=0):
        return _indent("// " + self.comment, indent)


class BlockdiagAttribute(BlockdiagElement):
    '''A key = value attribute element.'''

    def __init__(self, key, value):
        '''Creates a new key = value block diagram attribute

        Args:
            key: the attribute key
            value: the attribute value
        '''
        self.key = key
        self.value = value

    def render(self, indent=0):
        return _indent("%s = %s;" % (self.key, self.value), indent)


class BlockdiagNode(BlockdiagElement):
    '''A name [key = value, ...] node element.'''

    def __init__(self, name, *args, **kwargs):
        '''Creates a new block diagram node.

        Args:
            name: the name of the node
            *args: optional non-keyword arguments for the node
            **kwargs: optional key = value attributes for the node
        '''
        self.name = name
        self.args = list(args)
        self.kwargs = kwargs

    def render(self, indent=0):
        line = self.name
        attributes = self.args + [
            "%s = %s" % (k, v) for k, v in iteritems(self.kwargs)
        ]
        if attributes:
            line += " [" + ", ".join(attributes) + "];"
        return _indent(line, indent)


class BlockdiagDirectedEdge(BlockdiagElement):
    '''A source -> sink directed edge element.'''

    def __init__(self, source, sink):
        '''Creates a new block diagram directed edge.

        Args:
            source: the edge source
            sink: the edge sink
        '''
        self.source = source
        self.sink = sink

    def render(self, indent=0):
        return _indent(self.source + " -> " + self.sink + ";", indent)


def _indent(line, indent):
    return " " * indent + line


def _node_extent(line):
    return int(round(64 + 10 * len(line)))


def _module_extent(name):
    return int(round(max(8.5 * len(name), 50)))
