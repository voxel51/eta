'''
theta.modules.filter_labels
    -> threshold labels
    -> schema filter (KEEP or DELETE)

'''
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


def validate_input(func):

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper



# MANAGER


class LabelsTransformManager():
    '''
    - keeps track of a series of transforms
    - manages running transforms on Labels, SetLabels, Dataset, Directory or
        List of Labels
    '''

    def __init__(self):
        self._transforms = []

    def list_transforms(self):
        return [etau.get_class_name(x) for x in self._transforms]

    @validate_input
    def add_transform(self, labels_transform):
        self._transforms.append(labels_transform)



class LabelsTransform():
    @property
    def num_labels_transformed(self):
        return self._num_labels_transformed

    @property
    def report(self):
        return {
            "num_labels_transformed": self.num_labels_transformed,
        }

    def __init__(self, inplace=False):
        '''

        Args:
            inplace: if True, modify labels in place
        '''
        self._inplace = inplace
        self.clear_state()

    def clear_state(self):
        self._num_labels_transformed = 0

    def transform(self, labels, labels_path=None):
        raise NotImplementedError("Subclass must implement")


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
