'''
Tool for generating metadata JSON files programmatically for ETA modules.

Syntax:
    python generate_module_metadata.py /path/to/eta_module.py

Copyright 2018, Voxel51, LLC
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

from collections import defaultdict
import json
import logging
import os
import re
import sys

from docutils.core import publish_doctree
from docutils.nodes import paragraph, field_list, field_name, field_body

from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.util.docstrings import prepare_docstring

import eta.core.module as etam


logger = logging.getLogger(__name__)


def _parse_info_section(self, section):
    return self._x_parse_section("info", section)


def _parse_inputs_section(self, section):
    return self._x_parse_section("input", section)


def _parse_outputs_section(self, section):
    return self._x_parse_section("output", section)


def _parse_attributes_section(self, section):
    return self._x_parse_section("attribute", section)


def _parse_parameters_section(self, section):
    return self._x_parse_section("parameter", section)


def _parse_section(self, var, section):
    lines = []
    for _name, _type, _desc in self._consume_fields():
        field = ":%s %s: " % (var, _name)
        lines.extend(self._format_block(field, _desc))
        if _type:
            lines.append(":type %s: %s" % (_name, _type))
    lines.append("")
    return lines


def _parse(self):
    self._sections["info"] = self._x_parse_info_section
    self._sections["inputs"] = self._x_parse_inputs_section
    self._sections["outputs"] = self._x_parse_outputs_section
    self._sections["attributes"] = self._x_parse_attributes_section
    self._sections["parameters"] = self._x_parse_parameters_section
    self._x_original_parse()


GoogleDocstring._x_parse_section = _parse_section
GoogleDocstring._x_parse_info_section = _parse_info_section
GoogleDocstring._x_parse_inputs_section = _parse_inputs_section
GoogleDocstring._x_parse_outputs_section = _parse_outputs_section
GoogleDocstring._x_parse_attributes_section = _parse_attributes_section
GoogleDocstring._x_parse_parameters_section = _parse_parameters_section
GoogleDocstring._x_original_parse = GoogleDocstring._parse
GoogleDocstring._parse = _parse


class ModuleDocstring(object):
    '''Class encapsulating docstrings in ETA modules.'''

    def __init__(self, docstr):
        self.short_desc = ""
        self.long_desc = ""
        self.info = defaultdict(dict)
        self.attributes = defaultdict(dict)
        self.inputs = defaultdict(dict)
        self.outputs = defaultdict(dict)
        self.parameters = defaultdict(dict)

        self._last_dict = None
        self._parse_docstring(docstr)

    def to_dict(self):
        return {
            "short_desc": self.short_desc.strip(),
            "long_desc": self.long_desc.strip(),
            "info": dict(self.info),
            "attributes": dict(self.attributes),
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "parameters": dict(self.parameters),
        }

    def _parse_docstring(self, docstr):
        gds = GoogleDocstring(prepare_docstring(docstr))
        doctree = publish_doctree(str(gds))
        self._parse_doctree(doctree)

    def _parse_doctree(self, doctree):
        for node in doctree:
            if isinstance(node, paragraph):
                if not self.short_desc:
                    self.short_desc = node.astext()
                else:
                    self.long_desc += "\n\n" + node.astext()
            elif isinstance(node, field_list):
                for field in node:
                    self._parse_field(field)
            else:
                logger.info("Ignoring unsupported node '%s'" % node.astext())

    def _parse_field(self, field):
        # Parse field content
        name = _get_name(field)
        body = _get_body(field)
        try:
            meta, name = name.split(" ", 1)
        except ValueError:
            raise Exception("Invalid field '%s'" % field.astext())

        # Process based on meta tag
        if meta == "info":
            self.info[name] = body
        elif meta == "input":
            self._parse_node_body(self.inputs[name], body)
        elif meta == "output":
            self._parse_node_body(self.outputs[name], body)
        elif meta == "parameter":
            self._parse_parameter_body(self.parameters[name], body)
        elif meta == "attribute":
            self._parse_attribute_body(self.attributes[name], body)
        elif meta == "type":
            self._last_dict["type"] = body
        else:
            raise Exception("Invalid field '%s'" % field.astext())

    def _parse_node_body(self, d, body):
        body, default, required = _parse_default_element(body)
        if default:
            raise Exception(
                "Module inputs/outputs must have empty ('' or None) default "
                "values, but '%s' was found" % str(default)
            )
        d["description"] = body
        d["required"] = required
        self._last_dict = d

    def _parse_parameter_body(self, d, body):
        body, default, required = _parse_default_element(body)
        d["description"] = body
        d["default"] = default
        d["required"] = required
        self._last_dict = d

    def _parse_attribute_body(self, d, body):
        d["description"] = body
        self._last_dict = d


def _get_name(field):
    index = field.first_child_matching_class(field_name)
    if index is None:
        raise Exception("Expected field_name")
    return field.children[index].astext()


def _get_body(field):
    index = field.first_child_matching_class(field_body)
    if index is None:
        raise Exception("Expected field_body")
    return field.children[index].astext().replace("\n", " ").strip()


def _parse_default_element(body):
    m = re.search(r"\[(?P<default>\w*)\]", body)
    if m:
        try:
            raw = m.group("default")
            default = json.loads(raw)
        except ValueError:
            raise Exception(
                "Invalid default value '%s'. Remember that default values "
                "must be expressed as JSON, not Python values." % raw
            )
        body = body.replace(m.group(0), "")
        required = False
    else:
        default = None
        required = True

    return body.strip(), default, required

