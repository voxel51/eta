'''
theta.modules.filter_labels
    -> threshold labels
    -> schema filter (KEEP or DELETE)

'''
from collections import OrderedDict

import eta.core.datasets as etads
import eta.core.utils as etau

# rename labels
rename_labels_config = {
    "frames": {
        "TOD": "time of day",
        "time of day:dawn": "time of day:morning"
    },
    "objects": {
        "people": "person",
        "vehicle:pose:absolute rear": "vehicle:pose:rear",
        "*:color:porrple": "*:color:pUrple"
    }
}

# threshold labels
threshold_labels_config = {
    "frames": {
        # frame attribute
        "*": 0.1,
        "time of day": 0.2,
        "time of day:*": 0.4,
        "time of day:dawn": 0.5
    },
    "objects": {
        # object label
        "*": 0.1,       # filter all object labels
        "person": 0.2,  # filter person object label
        # object attributes
        "*:*:*": 0.1,               # filter all attributes
        "*:color:*": 0.2,           # filter all color attributes
        "*:color:red": 0.3,         # filter all color red attributes
        "vehicle:color:red": 0.4    # filter vehicle color red attributes
    }
}


RAISE = "RAISE"
SKIP = "SKIP"
OVERRIDE = "OVERRIDE"
COLLISION_HANDLE_OPTIONS = {RAISE, SKIP, OVERRIDE}


# MANAGER


class LabelsTransformManager():
    '''
    - keeps track of a series of transforms
    - manages running transforms on Labels, SetLabels, Dataset, Directory or
        List of Labels
    '''

    def __init__(self):
        self._transforms = OrderedDict()

    def list_transforms(self):
        return [etau.get_class_name(x) for x in self._transforms]

    def get_reports(self):
        reports = {}
        for k, transform in self._transforms.items():
            reports[k] = transform.report
        return reports

    def add_transform(self, labels_transform):
        if not isinstance(labels_transform, LabelsTransform):
            raise ValueError(
                "Unexpected type: '%s'".format(type(labels_transform)))

        k = str(len(self._transforms)) + " - " + \
                etau.get_class_name(labels_transform)

        self._transforms[k] = labels_transform

    def transform_labels(self, labels, labels_path=None):
        for transform in self._transforms.values():
            transform.transform(labels, labels_path=None)

        if labels_path:
            labels.write_json(labels_path)

    def transform_set_labels(
            self, set_labels, set_labels_path=None, verbose=20):
        for idx, labels in enumerate(set_labels):
            if verbose and idx % verbose == 0:
                print("%4d/%4d" % (idx, len(set_labels)))

            self.transform_labels(labels, labels_path=None)

        if set_labels_path:
            set_labels.write_json(set_labels_path)

    def transform_dataset(self, dataset: etads.LabeledDataset, verbose=20):
        for idx, labels_path in enumerate(dataset.iter_labels_paths()):
            if verbose and idx % verbose == 0:
                print("%4d/%4d" % (idx, len(dataset)))

            labels = dataset.read_labels(labels_path)

            self.transform_labels(labels, labels_path=labels_path)



class LabelsTransform():
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

    def clear_state(self):
        self._num_labels_transformed = 0

    def transform(self, labels, labels_path=None):
        self._num_labels_transformed += 1


# METADATA


'''
1) eta.core.datasets.standardize.ensure_labels_filename_property
'''
class EnsureFileName(LabelsTransform):
    '''Populates the labels.filename'''

    @property
    def num_populated(self):
        return self._num_populated

    @property
    def num_skipped(self):
        return self._num_skipped

    @property
    def num_overriden(self):
        return self._num_overriden

    def __init__(self, dataset, inplace=False, mismatch_handle=RAISE):
        super(EnsureFileName, self).__init__(inplace=inplace)
        self._num_populated = 0
        self._num_skipped = 0
        self._num_overriden = 0


# SCHEMA COMPARISON


'''
1) eta.core.image.ImageLabelsSyntaxChecker
   eta.core.video.VideoLabelsSyntaxChecker
'''
class MatchSyntax(LabelsTransform):
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


class MapLabels(LabelsTransform):
    '''Provided a mapping config, rename from one label string to another'''

    def __init__(self, rename_config, inplace=False):
        super(MapLabels, self).__init__(inplace=inplace)


'''
1) eta.core.datasets.standardize.check_duplicate_attrs
'''
class CheckExclusiveAttributes(LabelsTransform):
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
    def __init__(self, schema):
        super(CheckExclusiveAttributes, self).__init__()
        self._schema = schema

    def transform(self, labels, labels_path=None):
        super(CheckExclusiveAttributes, self).transform(
            labels, labels_path=labels_path)
        # raise NotImplementedError("TODO TYLER")

'''

'''
class CheckConstantAttributes(LabelsTransform):
    '''check for attributes that should not vary over time (video attrs,
    constant object attrs, constant event attrs...
    '''


class CheckAgainstSchema(LabelsTransform):
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

