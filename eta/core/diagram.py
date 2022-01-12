"""
Core tools for building block diagrams of modules and pipelines.

Block diagrams are built using the `blockdiag` package.

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
from future.utils import iteritems, itervalues

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import copy
import logging
import os

import eta.core.utils as etau


logger = logging.getLogger(__name__)


TOP_LEVEL_NAME = "blockdiag"
GROUP_NAME = "group"
INDENTATION = 2
PARAM_COLOR = '"#FF6D05"'  # Voxel51 orange
MODULE_COLOR = '"#F4F4F4"'  # Voxel51 gray
MODULE_HEIGHT = 60
NODE_HEIGHT = 40


class HasBlockDiagram(object):
    """Mixin class for objects that can be rendered as block diagrams."""

    def to_blockdiag(self):
        """Returns a BlockdiagFile representation of the instance.

        Returns:
             a BlockdiagFile instance
        """
        raise NotImplementedError("subclass must implement to_blockdiag()")

    def render(self, path, keep_diag_file=False):
        """Renders the block diagram as an SVG image.

        Args:
            path: the output path, which must have extension ".svg"
            keep_diag_file: whether to keep the .diag file (True) or delete it
                after the SVG is generated (False). The default is False
        """
        # Parse path
        base, ext = os.path.splitext(path)
        blockdiag_path = base + ".diag"
        svg_path = base + ".svg"
        if ext != ".svg":
            logger.warning("Replacing extension of '%s' with '.svg'", path)

        # Generate .diag
        if keep_diag_file:
            logger.info("Generating blockdiag file '%s'", blockdiag_path)
        self.to_blockdiag().write(blockdiag_path)

        try:
            # Generate .svg
            logger.info("Generating block diagram '%s'", svg_path)
            args = ["blockdiag", "-Tsvg", "-o", svg_path, blockdiag_path]
            etau.communicate_or_die(args)
        except etau.ExecutableNotFoundError:
            raise etau.PackageError(
                "You must run pip install voxel51-eta[pipeline] in order to "
                "use this feature"
            )
        finally:
            if not keep_diag_file:
                etau.delete_file(blockdiag_path)


class BlockdiagFile(object):
    """Base class for representing a block diagram in the blockdiag syntax."""

    def __init__(self):
        """Creates a BlockdiagFile instance."""
        self._file = BlockdiagGroup(is_top_level=True)

    def add_element(self, element):
        """Adds the element to the file.

        Args:
            element: a BlockdiagElement
        """
        self._file.add_element(element)

    def export(self, **kwargs):
        """Exports the file as a BlockdiagGroup.

        Args:
            **kwargs: optional attributes to add to the group

        Returns:
            a BlockdiagGroup describing the file
        """
        group = copy.deepcopy(self._file)
        group.name = GROUP_NAME  # downgrade from top-level to nested group
        group.add_attributes(**kwargs)
        return group

    def write(self, path):
        """Writes the block diagram to disk as a `.diag` text file.

        The output directory is created, if necessary.

        Args:
            path: the output path
        """
        diagram_str = self._file.render()
        etau.write_file(diagram_str, path)


class BlockdiagPipeline(BlockdiagFile):
    """Class representing a pipeline as a blockdiag file."""

    def __init__(self, name):
        """Creates a BlockdiagPipeline instance.

        Args:
            name: the name of the pipeline
        """
        self.name = name
        super(BlockdiagPipeline, self).__init__()

        self._inputs = BlockdiagSection(comment="inputs")
        self._outputs = BlockdiagSection(comment="outputs")
        self._connections = BlockdiagSection(comment="connections")
        self._modules = BlockdiagSection(comment="modules")
        self._module_prefixes = {}

        self.add_element(self._inputs)
        self.add_element(self._outputs)
        self.add_element(self._connections)
        self.add_element(self._modules)

    def add_module(self, name, blockdiag_module):
        """Adds a module to the pipeline.

        Args:
            name: the name of the module in the pipeline
            blockdiag_module: a BlockdiagModule instance
        """
        group = blockdiag_module.export(color=MODULE_COLOR)
        # Prefix the module names to ensure uniqueness between modules
        group.prefix_names(self._get_prefix(name))
        self._modules.add_element(group)

    def add_input(self, name):
        """Adds an input to the pipeline.

        Args:
            name: the input name
        """
        node = BlockdiagNode(
            name, shape="cloud", height=NODE_HEIGHT, width=_node_extent(name)
        )
        self._inputs.add_element(node)

    def add_output(self, name):
        """Adds an output to the pipeline.

        Args:
            name: the output name
        """
        node = BlockdiagNode(
            name, shape="cloud", height=NODE_HEIGHT, width=_node_extent(name)
        )
        self._outputs.add_element(node)

    def add_connection(self, source, sink):
        """Adds a connection between the given nodes.

        Args:
            source: the source PipelineNode
            sink: the sink PipelineNode
        """
        src = source.node
        snk = sink.node
        if source.is_module_input or source.is_module_output:
            src = self._get_prefix(source.module) + src
        if sink.is_module_input or sink.is_module_output:
            snk = self._get_prefix(sink.module) + snk
        edge = BlockdiagDirectedEdge(src, snk)
        self._connections.add_element(edge)

    def _get_prefix(self, name):
        try:
            prefix = self._module_prefixes[name]
        except KeyError:
            prefix = "%d." % (len(self._module_prefixes) + 1)
            self._module_prefixes[name] = prefix

        return prefix


class BlockdiagModule(BlockdiagFile):
    """Class representing a module as a blockdiag file."""

    def __init__(self, name):
        """Creates a BlockdiagModule instance.

        Args:
            name: the name of the module
        """
        self.name = name
        super(BlockdiagModule, self).__init__()

        self._module = BlockdiagSection(comment="module")
        self._inputs = BlockdiagSection(comment="inputs")
        self._outputs = BlockdiagSection(comment="outputs")
        self._parameters = BlockdiagSection(comment="parameters")
        self._io_edges = BlockdiagSection(comment="I/O connections")
        self._param_edges = BlockdiagSection(comment="parameter connections")

        self._add_module(name)

        self.add_element(self._module)
        self.add_element(self._inputs)
        self.add_element(self._outputs)
        self.add_element(self._parameters)
        self.add_element(self._io_edges)
        self.add_element(self._param_edges)

    def add_input(self, name):
        """Add a module input with the given name.

        Args:
            name: the name of the module input
        """
        node = BlockdiagNode(
            name,
            shape="endpoint",
            height=NODE_HEIGHT,
            width=_node_extent(name),
        )
        self._inputs.add_element(node)
        self._add_io_edge(name, self.name)

    def add_output(self, name):
        """Add a module output with the given name.

        Args:
            name: the name of the module output
        """
        node = BlockdiagNode(
            name,
            shape="endpoint",
            height=NODE_HEIGHT,
            width=_node_extent(name),
        )
        self._outputs.add_element(node)
        self._add_io_edge(self.name, name)

    def add_parameter(self, name):
        """Add a module parameter with the given name.

        Args:
            name: the name of the module parameter
        """
        node = BlockdiagNode(
            name,
            shape="beginpoint",
            rotate=270,
            height=_node_extent(name),
            width=NODE_HEIGHT,
        )
        self._parameters.add_element(node)
        self._add_param_edge(name)

    def _add_module(self, name):
        node = BlockdiagNode(
            name, shape="box", height=MODULE_HEIGHT, width=_module_extent(name)
        )
        self._module.add_element(node)
        self._param_group = BlockdiagGroup(
            orientation="portrait", color=PARAM_COLOR
        )
        self._param_edges.add_element(self._param_group)

    def _add_io_edge(self, source, sink):
        edge = BlockdiagDirectedEdge(source, sink)
        self._io_edges.add_element(edge)

    def _add_param_edge(self, param_name):
        edge = BlockdiagDirectedEdge(param_name, self.name)
        self._param_group.add_element(edge)


class BlockdiagElement(object):
    """Base class for all elements of a block diagram file."""

    def render(self, indent=0):
        """Renders the element as a string at the given indentiation level.

        Args:
            indent: the indentation level at which to render the element

        Returns:
            a string containing the rendered element
        """
        raise NotImplementedError("subclass must implement render()")


class BlockdiagContainer(BlockdiagElement):
    """Base class for block digram elements that contain other elements."""

    def __init__(self):
        """Creates a BlockdiagContainer instance."""
        self.elements = []

    @property
    def is_empty(self):
        """Returns True/False if the container is empty."""
        return len(self.elements) == 0

    def prefix_names(self, prefix):
        """Prepend the given prefix to all nodes names in the given container.

        This function operates recursively on nested containers.

        Args:
            prefix: the string to prepend
        """
        for e in self.elements:
            if isinstance(e, BlockdiagContainer):
                e.prefix_names(prefix)
            elif isinstance(e, BlockdiagNode):
                e.name = prefix + e.name
            elif isinstance(e, BlockdiagDirectedEdge):
                e.source = prefix + e.source
                e.sink = prefix + e.sink

    def add_element(self, element):
        """Adds the element to the container.

        Args:
            element: a BlockdiagElement
        """
        self.elements.append(element)


class BlockdiagGroup(BlockdiagContainer):
    """A group of BlockdiagElements."""

    def __init__(self, is_top_level=False, **kwargs):
        """Creates a BlockdiagGroup instance.

        Args:
            is_top_level: whether this is the top-level blockdiag group or a
                nested group. By default, this is False
            **kwargs: optional attributes for the group
        """
        super(BlockdiagGroup, self).__init__()
        self.name = TOP_LEVEL_NAME if is_top_level else GROUP_NAME
        self.attributes = []
        self.add_attributes(**kwargs)

    def add_attributes(self, **kwargs):
        """Adds the given attributes to the group.

        Args:
            **kwargs: the attributes to add to the group
        """
        for k, v in iteritems(kwargs):
            self.attributes.append(BlockdiagAttribute(k, v))

    def render(self, indent=0):
        """Renders the BlockdiagGroup as a string with the given indentation.

        Args:
            indent: the indentation level at which to render the element

        Returns:
            a string containing the rendered BlockdiagGroup
        """
        if self.is_empty:
            return ""
        c = [_indent(self.name + " {", indent)]
        for a in self.attributes:
            c.append(a.render(indent=indent + INDENTATION))
        for e in self.elements:
            c.append(e.render(indent=indent + INDENTATION))
        c.append(_indent("}", indent))
        return "\n".join(c).rstrip()


class BlockdiagSection(BlockdiagContainer):
    """A list of BlockdiagElements with an optional BlockdiagComment above."""

    def __init__(self, comment=None):
        """Creates a BlockdiagSection instance.

        Args:
            comment: an optional comment string
        """
        super(BlockdiagSection, self).__init__()
        if comment:
            self.add_element(BlockdiagComment(comment))

    def render(self, indent=0):
        """Renders the BlockdiagSection as a string with the given indentation.

        Args:
            indent: the indentation level at which to render the element

        Returns:
            a string containing the rendered BlockdiagSection
        """
        c = []
        for e in self.elements:
            c.append(e.render(indent=indent))
        content = "\n".join(c).rstrip()
        return ("\n" if content else "") + content


class BlockdiagComment(BlockdiagElement):
    """A BlockdiagElement representing a comment."""

    def __init__(self, comment):
        """Creates a BlockdiagComment instance.

        Args:
            comment: a comment string
        """
        self.comment = comment

    def render(self, indent=0):
        """Renders the BlockdiagComment as a string with the given indentation.

        Args:
            indent: the indentation level at which to render the element

        Returns:
            a string containing the rendered BlockdiagComment
        """
        return _indent("// " + self.comment, indent)


class BlockdiagAttribute(BlockdiagElement):
    """A BlockdiagElement representing a key-value pair."""

    def __init__(self, key, value):
        """Creates a BlockdiagAttribute instance.

        Args:
            key: the attribute key
            value: the attribute value
        """
        self.key = key
        self.value = value

    def render(self, indent=0):
        """Renders the BlockdiagAttribute as a string with the given
        indentation.

        Args:
            indent: the indentation level at which to render the element

        Returns:
            a string containing the rendered BlockdiagAttribute
        """
        return _indent("%s = %s;" % (self.key, self.value), indent)


class BlockdiagNode(BlockdiagElement):
    """A BlockdiagElement representing a `name [key = value, ...]` node."""

    def __init__(self, name, *args, **kwargs):
        """Creates a BlockdiagNode instance.

        Args:
            name: the name of the node
            *args: optional non-keyword arguments for the node
            **kwargs: optional attributes for the node
        """
        self.name = name
        self.args = list(args)
        self.kwargs = kwargs

    def render(self, indent=0):
        """Renders the BlockdiagNode as a string with the given indentation.

        Args:
            indent: the indentation level at which to render the element

        Returns:
            a string containing the rendered BlockdiagNode
        """
        line = self.name
        attributes = self.args + [
            "%s = %s" % (k, v) for k, v in iteritems(self.kwargs)
        ]
        if attributes:
            line += " [" + ", ".join(attributes) + "];"
        return _indent(line, indent)


class BlockdiagDirectedEdge(BlockdiagElement):
    """A BlockdiagElement representing a `source -> sink` directed edge."""

    def __init__(self, source, sink):
        """Creates a BlockdiagDirectedEdge instance.

        Args:
            source: the edge source
            sink: the edge sink
        """
        self.source = source
        self.sink = sink

    def render(self, indent=0):
        """Renders the BlockdiagDirectedEdge as a string with the given
        indentation.

        Args:
            indent: the indentation level at which to render the element

        Returns:
            a string containing the rendered BlockdiagDirectedEdge
        """
        return _indent(self.source + " -> " + self.sink + ";", indent)


def _indent(line, indent):
    return " " * indent + line


def _node_extent(line):
    return int(round(64 + 10 * len(line)))


def _module_extent(name):
    return int(round(max(8.5 * len(name), 50)))
