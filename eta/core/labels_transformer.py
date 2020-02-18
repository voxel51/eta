'''
TODO
theta.modules.filter_labels
    -> threshold labels
    -> schema filter (KEEP or DELETE)

Copyright 2017-2020, Voxel51, Inc.
voxel51.com

Tyler Ganter, tyler@voxel51.com
'''
from collections import OrderedDict, Counter, defaultdict
from copy import deepcopy
import logging

import eta.core.data as etad
import eta.core.datasets as etads
import eta.core.utils as etau
import eta.core.video as etav


logger = logging.getLogger(__name__)


# MANAGER


class LabelsTransformerManager():
    '''
    - keeps track of a series of transforms
    - manages running transforms on Labels, SetLabels, Dataset, Directory or
        List of Labels
    '''

    def __init__(self):
        self._transforms = OrderedDict()

    def list_transformers(self):
        return [etau.get_class_name(x) for x in self._transforms]

    def get_reports(self):
        reports = {}
        for k, transform in self._transforms.items():
            reports[k] = transform.report
        return reports

    def add_transformer(self, transformer):
        if not isinstance(transformer, LabelsTransformer):
            raise ValueError(
                "Unexpected type: '%s'".format(type(transformer)))

        k = str(len(self._transforms)) + " - " + \
            etau.get_class_name(transformer)

        self._transforms[k] = transformer

    def transform_labels(self, labels, labels_path=None):
        for transform in self._transforms.values():
            transform.transform(labels, labels_path=None)

        if labels_path:
            labels.write_json(labels_path)

    def transform_set_labels(
            self, set_labels, set_labels_path=None, verbose=20):
        for idx, labels in enumerate(set_labels):
            if verbose and idx % verbose == 0:
                logger.info("%4d/%4d" % (idx, len(set_labels)))

            self.transform_labels(labels, labels_path=None)

        if set_labels_path:
            set_labels.write_json(set_labels_path)

    def transform_dataset(self, dataset: etads.LabeledDataset, verbose=20):
        for idx, labels_path in enumerate(dataset.iter_labels_paths()):
            if verbose and idx % verbose == 0:
                logger.info("%4d/%4d" % (idx, len(dataset)))

            labels = dataset.read_labels(labels_path)

            self.transform_labels(labels, labels_path=labels_path)

            # break # @todo(Tyler) TEMP


# ABSTRACT CLASS


class LabelsTransformer():
    @property
    def num_labels_transformed(self):
        return self._num_labels_transformed

    @property
    def report(self):
        return {
            "num_labels_transformed": self.num_labels_transformed,
        }

    def __init__(self):
        self.clear_state()

        if type(self) == LabelsTransformer:
            raise TypeError("Cannot instantiate abstract class %s"
                            % etau.get_class_name(LabelsTransformer))

    def clear_state(self):
        self._num_labels_transformed = 0

    def transform(self, labels, labels_path=None):
        self._num_labels_transformed += 1


class LabelsTransformerError(Exception):
    '''Error raised when a LabelsTransformer is violated.'''
    pass


# TRANSFORMERS


class LabelsMapper(LabelsTransformer):
    pass


class SyntaxChecker(LabelsTransformer):
    pass


class SchemaFilter(LabelsTransformer):
    pass


class ConfidenceThresholder(LabelsTransformer):
    pass


# TODO

'''
1) eta.core.image.ImageLabelsSyntaxChecker
   eta.core.video.VideoLabelsSyntaxChecker
'''
class MatchSyntax(LabelsTransformer):
    '''Using a target schema, match capitalization and underscores versus spaces
    to match the schema
    '''

    @property
    def target_schema(self):
        return self._target_schema

    @property
    def fixable_schema(self):
        return self._fixable_schema

    @property
    def unfixable_schema(self):
        return self._unfixable_schema

    def __init__(self, target_schema, inplace=False):
        super(MatchSyntax, self).__init__(inplace=inplace)
        self._target_schema = target_schema
        self._fixable_schema = None
        self._unfixable_schema = None


class MapLabels(LabelsTransformer):
    '''Provided a mapping config, rename from one label string to another'''

    def __init__(self, rename_config, inplace=False):
        super(MapLabels, self).__init__(inplace=inplace)


'''
1) eta.core.datasets.standardize.check_duplicate_attrs
'''
class CheckExclusiveAttributes(LabelsTransformer):
    '''duplicate exclusive attributes (exclusive attributes can only have one
    value, such as person:sex

    Cases:
        same value:
            "male", "male"
            delete one, delete both, or raise error
        different values:
            "male", "female"
            delete both, or raise error
    '''
    @property
    def attrs_deduplicated(self):
        return self._attrs_deduplicated

    @property
    def attrs_voted(self):
        return self._attrs_voted

    @property
    def attrs_removed(self):
        return self._attrs_removed

    @property
    def min_agreement(self):
        return self._min_agreement

    @property
    def report(self):
        d = super(CheckExclusiveAttributes, self).report
        d["attrs_deduplicated"] = self._attrs_deduplicated
        d["attrs_voted"] = self.attrs_voted
        d["attrs_removed"] = self.attrs_removed
        return d

    def __init__(self, schema, min_agreement=0):
        super(CheckExclusiveAttributes, self).__init__()
        self._schema = schema

        if min_agreement < 0 or min_agreement > 1:
            raise ValueError("min_agreement outside bounds [0, 1]")
        self._min_agreement = min_agreement

    def clear_state(self):
        super(CheckExclusiveAttributes, self).clear_state()
        self._attrs_deduplicated = defaultdict(int)
        self._attrs_voted = defaultdict(int)
        self._attrs_removed = defaultdict(int)

    def transform(self, labels, labels_path=None):
        super(CheckExclusiveAttributes, self).transform(
            labels, labels_path=labels_path)

        for attrs_schema, attrs in self._schema.iter_attr_containers(labels):
            self._transform_container(attrs, attrs_schema)

    def _transform_container(
            self, attrs: etad.AttributeContainer,
            schema: etad.AttributeContainerSchema,
    ):
        for attr_schema in schema.schema.values():
            if attr_schema.exclusive:
                self._transform_exclusive_attr(attrs, attr_schema)
            else:
                self._transform_nonexclusive_attr(attrs, attr_schema)

    def _transform_nonexclusive_attr(
            self, attrs: etad.AttributeContainer,
            schema: etad.AttributeSchema,
    ):
        '''this attribute is non-exclusive, but we still want to remove
        duplicates with the same `name:value` pair
        '''
        del_inds = set()
        found_values = set()

        for idx, attr in enumerate(attrs):
            if attr.name != schema.name:
                continue

            if attr.value in found_values:
                del_inds.add(idx)
                self._attrs_deduplicated[schema.name] += 1

            else:
                found_values.add(attr.value)

        attrs.delete_inds(del_inds)

    def _transform_exclusive_attr(
            self, attrs: etad.AttributeContainer,
            schema: etad.AttributeSchema,
    ):
        '''count all instances of attribute name and vote over values, choosing
        to keep a value only if there is sufficient "agreement"
        '''
        matching_attrs = attrs.get_attrs_with_name(schema.name)

        if len(matching_attrs) <= 1:
            return

        counter = Counter(x.value for x in matching_attrs)
        most_common_val, count = counter.most_common(1)[0]

        if count / len(matching_attrs) > self._min_agreement:
            # find the attr voted to keep
            self._attrs_voted[schema.name] += 1
            for attr in matching_attrs:
                if attr.value == most_common_val:
                    keep = attr
                    break
        else:
            # do not keep any attr with this name
            self._attrs_removed[schema.name] += 1
            keep = None

        attrs.filter_elements(filters=[
            lambda el: el == keep or el not in matching_attrs
        ])


'''

'''
class CheckConstantAttributes(LabelsTransformer):
    '''check for attributes that should not vary over time (video attrs,
    constant object attrs, constant event attrs...

    video attrs, check schema and raise error
    frame attrs, check schema and raise error

    Object attrs, TODO

    DetectedObjectAttrs, TODO
        collect all detected objects by ID
        for every constant attr in schema, check objects

    Event Attrs, TODO

    '''

    @property
    def attrs_populated(self):
        return self._attrs_populated

    @property
    def attrs_removed(self):
        return self._attrs_removed

    @property
    def min_agreement(self):
        return self._min_agreement

    @property
    def report(self):
        d = super(CheckConstantAttributes, self).report

        d["attrs_populated"] = {}
        for k in sorted(self.attrs_populated.keys()):
            d["attrs_populated"][k] = self.attrs_populated[k]

        d["attrs_removed"] = {}
        for k in sorted(self.attrs_removed.keys()):
            d["attrs_removed"][k] = self.attrs_removed[k]

        return d

    def __init__(self, schema, min_agreement=0.5):
        if not isinstance(schema, etav.VideoLabelsSchema):
            raise ValueError("Constant attributes only apply to %s"
                             % etau.get_class_name(etav.VideoLabelsSchema))

        super(CheckConstantAttributes, self).__init__()
        self._schema = schema

        if min_agreement < 0 or min_agreement > 1:
            raise ValueError("min_agreement outside bounds [0, 1]")
        self._min_agreement = min_agreement

    def clear_state(self):
        super(CheckConstantAttributes, self).clear_state()
        self._attrs_populated = defaultdict(int)
        self._attrs_removed = defaultdict(int)

    def transform(self, labels: etav.VideoLabels, labels_path=None):
        super(CheckConstantAttributes, self).transform(
            labels, labels_path=labels_path)

        if not isinstance(labels, etav.VideoLabels):
            raise ValueError("Constant attributes only apply to %s"
                             % etau.get_class_name(etav.VideoLabels))

        objects = self._collect_objects(labels.iter_objects())

        for (obj_label, obj_idx), obj_list in objects.items():
            for attr_schema in self._schema.objects[obj_label].schema.values():
                if attr_schema.constant:
                    self._check_constant(attr_schema, obj_list)

    def _collect_objects(self, obj_iterator):
        objects = defaultdict(list)

        for obj in obj_iterator:
            if obj.index is not None:
                objects[(obj.label, obj.index)].append(obj)

        return objects

    def _check_constant(self, schema: etad.AttributeSchema, obj_list):
        counter = Counter()
        for obj in obj_list:
            counter.update(
                set(obj.attrs.get_attr_values_with_name(schema.name)))

        for idx, (value, count) in enumerate(counter.most_common(len(counter))):
            # keep any values with agreement above min
            keep = count / len(obj_list) > self.min_agreement
            # but only keep values after the first most common if not exclusive
            keep &= (idx == 0 or not schema.exclusive)
            if keep:
                if count < len(obj_list):
                    self._populate_missing_attr(schema, value, obj_list)
            else:
                self._delete_attr(schema, value, obj_list)

    def _populate_missing_attr(
            self, schema: etad.AttributeSchema, attr_value, obj_list: list):
        '''For every DetectedObject in obj_list, add a copy of the attribute
        with the target attr_value from the nearest adjacent DetectedObject
        '''
        key = "%s:%s:%s" % (obj_list[0].label, schema.name, attr_value)
        self.attrs_populated[key] += 1

        for idx, obj in enumerate(obj_list):
            attr_vals = obj.attrs.get_attr_values_with_name(schema.name)
            if attr_value in attr_vals:
                continue

            left_list = obj_list[idx-1::-1]
            right_list = obj_list[idx+1::]
            left_list += [None] * (len(right_list) - len(left_list))
            right_list += [None] * (len(left_list) - len(right_list))
            match_obj = None

            for left_obj, right_obj in zip(left_list, right_list):
                if left_obj:
                    # check nearest preceding DetectedObject
                    cur_attr_vals = left_obj.attrs.get_attr_values_with_name(
                        schema.name)
                    if attr_value in cur_attr_vals:
                        match_obj = left_obj
                        break

                if right_obj:
                    # check nearest following DetectedObject
                    cur_attr_vals = right_obj.attrs.get_attr_values_with_name(
                        schema.name)
                    if attr_value in cur_attr_vals:
                        match_obj = right_obj
                        break

            if match_obj is None:
                raise LabelsTransformerError("This is not good...")

            attr_to_add = \
                deepcopy(match_obj.attrs.get_attrs_with_name(schema.name)[0])

            # if exclusive, replace the attr value instead of adding
            if schema.exclusive:
                obj.attrs.filter_elements(filters=[
                    lambda el: el.name != schema.name])

            obj.add_attribute(attr_to_add)

    def _delete_attr(
            self, schema: etad.AttributeSchema, attr_value, obj_list: list):
        key = "%s:%s:%s" % (obj_list[0].label, schema.name, attr_value)
        self.attrs_removed[key] += 1
        for obj in obj_list:
            obj.attrs.filter_elements(filters=[
                lambda el: el.name != schema.name or el.value != attr_value])



class CheckAgainstSchema(LabelsTransformer):
    '''
    - check against everything other than exclusive and constant
    - Filter anything that is not in the schema
    '''

    @property
    def filtered_schema(self):
        return self._filtered_schema

    def __init__(self, schema, keep=True, inplace=False):
        super(FilterBySchema, self).__init__(inplace=inplace)
        self._filtered_schema = None

