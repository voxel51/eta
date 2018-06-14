'''
Tool for generating metadata JSON files for ETA modules programmatically.

This script generates a module metadata JSON file in the same directory as the
input module file.

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
    '''Class encapsulating docstrings in ETA modules.

    This class uses `sphinx-napoleon` to parse Google style docstrings with
    `Info`, `Inputs`, `Outputs`, `Parameters`, and `Attributes` sections.

    Attributes:
        short_desc: the short description from the docstring (if any)
        long_desc: the long description from the docstring (if any)
        info: a dict of values specified in an `Info` section
        inputs: a dictionary of dicts describing any fields provided in an
            `Inputs` section
        outputs: a dictionary of dicts describing any fields provided in an
            `Outputs` section
        parameters: a dictionary of dicts describing any fields provided in an
            `Parameters` section
        attributes: a dictionary of dicts describing any fields provided in an
            `Attributes` section
    '''

    def __init__(self, docstr):
        self.short_desc = ""
        self.long_desc = ""
        self.info = defaultdict(lambda: None)
        self.inputs = defaultdict(lambda: defaultdict(lambda: None))
        self.outputs = defaultdict(lambda: defaultdict(lambda: None))
        self.parameters = defaultdict(lambda: defaultdict(lambda: None))
        self.attributes = defaultdict(lambda: defaultdict(lambda: None))

        self._last_dict = None
        self._parse_docstring(docstr)

    def _parse_docstring(self, docstr):
        gds = GoogleDocstring(prepare_docstring(docstr))
        doctree = publish_doctree(str(gds))
        self._parse_doctree(doctree)

    def _parse_doctree(self, doctree):
        for node in doctree:
            if isinstance(node, paragraph):
                pstr = node.astext().strip()
                if not self.short_desc:
                    self.short_desc = pstr
                elif not self.long_desc:
                    self.long_desc = pstr
                else:
                    self.long_desc += "\n\n" + pstr
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
            raise ModuleDocstringError(
                "Unsupported field '%s'" % field.astext())

        # Process based on meta tag
        if meta == "info":
            self._parse_info_body(self.info, name, body)
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
            raise ModuleDocstringError("Unsupported meta tag '%s'" % meta)

    def _parse_info_body(self, d, name, body):
        d[name] = body
        self._last_dict = {}

    def _parse_node_body(self, d, body):
        body, default, required = _parse_default_element(body)
        if default:
            raise ModuleDocstringError(
                "Module inputs/outputs must have empty ('' or None) default "
                "values, but '%s' was found" % str(default))
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


class ModuleDocstringError(Exception):
    '''Error raised when there was a problem parsing a module docstring.'''
    pass


class ModuleMetadataGenerationError(Exception):
    '''Error raised when there was a problem generating the module metadata
    file for an ETA module.
    '''
    pass


def _get_name(field):
    index = field.first_child_matching_class(field_name)
    if index is None:
        raise ModuleDocstringError(
            "Expected field_name in field: %s" % field.astext())
    return field.children[index].astext()


def _get_body(field):
    index = field.first_child_matching_class(field_body)
    if index is None:
        raise ModuleDocstringError(
            "Expected field_body in field: %s" % field.astext())
    return field.children[index].astext().replace("\n", " ").strip()


def _parse_default_element(body):
    m = re.search(r"\[(?P<default>\w*)\]", body)
    if m:
        try:
            raw = m.group("default")
            default = json.loads(raw)
        except ValueError:
            raise ModuleDocstringError(
                "Invalid default value '%s'. Remember that default values "
                "must be expressed as JSON, not Python values." % raw)
        body = body.replace(m.group(0), "")
        required = False
    else:
        default = None
        required = True

    return body.strip(), default, required


def _get_module_docstring(module_name):
    __import__(module_name)
    return sys.modules[module_name].__doc__


def _get_module_config_docstring(module_name):
    __import__(module_name)
    for cls in itervalues(sys.modules[module_name].__dict__):
        if isinstance(cls, type) and issubclass(cls, etam.BaseModuleConfig):
            logger.info("Found module config class '%s'", cls.__name__)
            return cls.__doc__

    raise Exception("No subclass of BaseModuleConfig found")


def _get_class_docstring(module_name, class_name):
    return getattr(sys.modules[module_name], class_name).__doc__


def _build_module_metadata(module_name, mds, dds, pds):
    try:
        logger.info("*** Building info field")
        info_builder = (etam.ModuleInfoConfig.builder()
            .set(name=module_name)
            .set(type=mds.info["type"])
            .set(version=mds.info["version"])
            .set(description=mds.short_desc.rstrip("?:!.,;"))
            .set(exe=module_name + ".py")
            .validate())
    except Exception as e:
        raise ModuleMetadataGenerationError(
            "Error populating the 'info' field:\n" + str(e))

    try:
        logger.info("*** Building inputs")
        inputs = []
        for iname, ispec in iteritems(dds.inputs):
            node_builder = (etam.ModuleNodeConfig.builder()
                .set(name=iname)
                .set(type=ispec["type"])
                .set(description=ispec["description"])
                .set(required=ispec["required"])
                .validate())
            inputs.append(node_builder)
    except Exception as e:
        raise ModuleMetadataGenerationError(
            "Error populating the 'inputs' field:\n" + str(e))

    try:
        logger.info("*** Building outputs")
        outputs = []
        for oname, ospec in iteritems(dds.outputs):
            node_builder = (etam.ModuleNodeConfig.builder()
                .set(name=oname)
                .set(type=ospec["type"])
                .set(description=ospec["description"])
                .set(required=ospec["required"])
                .validate())
            outputs.append(node_builder)
    except Exception as e:
        raise ModuleMetadataGenerationError(
            "Error populating the 'outputs' field:\n" + str(e))

    try:
        logger.info("*** Building parameters")
        parameters = []
        for pname, pspec in iteritems(pds.parameters):
            parameter_builder = (etam.ModuleParameterConfig.builder()
                .set(name=pname)
                .set(type=pspec["type"])
                .set(description=pspec["description"])
                .set(required=pspec["required"])
                .set(default=pspec["default"])
                .validate())
            parameters.append(parameter_builder)
    except Exception as e:
        raise ModuleMetadataGenerationError(
            "Error populating the 'parameters' field:\n" + str(e))

    try:
        logger.info("*** Building ModuleMetadataConfig")
        mmc = (etam.ModuleMetadataConfig.builder()
            .set(info=info_builder)
            .set(inputs=inputs)
            .set(outputs=outputs)
            .set(parameters=parameters)
            .build())
    except Exception as e:
        raise ModuleMetadataGenerationError(
            "Error building the ModuleMetadataConfig:\n" + str(e))

    return mmc


def main(module_path):
    '''Generates the module metadata JSON file for the given ETA module.

    Args:
        module_path: the path to the .py file defining an ETA module
    '''
    module_dir, module_file = os.path.split(module_path)
    module_name = os.path.splitext(module_file)[0]
    sys.path.insert(0, module_dir)

    logger.info("Parsing module docstring")
    module_docstr = _get_module_docstring(module_name)
    mds = ModuleDocstring(module_docstr)

    logger.info("Parsing module config class docstring")
    config_docstr = _get_module_config_docstring(module_name)
    cds = ModuleDocstring(config_docstr)

    data_class = cds.attributes["data"]["type"]
    if not data_class:
        raise ModuleMetadataGenerationError(
            "Failed to find a data config class")
    logger.info("Parsing data class '%s' docstring", data_class)
    data_docstr = _get_class_docstring(module_name, data_class)
    dds = ModuleDocstring(data_docstr)

    parameters_class = cds.attributes["parameters"]["type"]
    if not parameters_class:
        raise ModuleMetadataGenerationError(
            "Failed to find a parameters config class")
    logger.info("Parsing parameter class '%s' docstring", parameters_class)
    parameters_doctstr = _get_class_docstring(module_name, parameters_class)
    pds = ModuleDocstring(parameters_doctstr)

    logger.info("Building module metadata")
    mmc = _build_module_metadata(module_name, mds, dds, pds)

    logger.info("Validating module metadata")
    etam.ModuleMetadata(mmc)

    outpath = os.path.join(module_dir, module_name + ".json")
    logger.info("Writing module metadata to %s", outpath)
    mmc.write_json(outpath)


if __name__ == "__main__":
    main(sys.argv[1])
