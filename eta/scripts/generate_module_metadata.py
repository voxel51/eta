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
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

from sphinx.ext.napoleon.docstring import GoogleDocstring


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

