'''
Core status infrastructure for pipelines and jobs.

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
from future.utils import iteritems
# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging

from eta.core.serial import Serializable
import eta.core.utils as etau


logger = logging.getLogger(__name__)


class PipelineStatus(Serializable):
    '''Class for recording the status of a pipeline.

    Attributes:
        name: the name of the pipeline
        status: the current status of the pipeline
        start_time: the time the pipeline was started, or None if not started
        complete_time: the time the pipeline was completed, or None if not
            completed
        fail_time: the time the pipeline failed, or None if not failed
        messages: a list of StatusMessage objects listing the status updates
            for the pipeline
        jobs: a list of JobStatus objects describing the status of the jobs
            that make up the pipeline
    '''

    # Pipeline status enum
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"

    def __init__(self, name, callback=None):
        '''Construct a new PipelineStatus instance.

        Args:
            name: the name of the pipeline
            callback: an optional callback function to invoke after each time
                the `write_json()` method is invoked. By default, no callback
                is invoked
        '''
        self.name = name
        self.status = PipelineStatus.QUEUED
        self.start_time = None
        self.complete_time = None
        self.fail_time = None
        self.messages = []
        self.jobs = []

        self._callback = callback
        self._active_job = None

    def attributes(self):
        return [
            "name", "status", "start_time", "complete_time", "fail_time",
            "messages", "jobs",
        ]

    @property
    def active_job(self):
        '''The JobStatus instance for the active job, or None if no job is
        active.
        '''
        return self._active_job

    def add_job(self, name):
        '''Add a new job with the given name and activate it.

        Returns:
            the JobStatus instance for the job
        '''
        self._active_job = JobStatus(name)
        self.jobs.append(self._active_job)
        return self._active_job

    def add_message(self, message):
        '''Add the given message to the messages list.'''
        status_message = StatusMessage(message)
        self.messages.append(status_message)
        return status_message.time

    def start(self, message="Pipeline started"):
        '''Mark the pipeline as started and record the given message.'''
        self.start_time = self.add_message(message)
        self.status = PipelineStatus.RUNNING

    def complete(self, message="Pipeline completed"):
        '''Mark the pipelne as complete and record the given message.'''
        self.complete_time = self.add_message(message)
        self.status = PipelineStatus.COMPLETE

    def fail(self, message="Pipeline failed"):
        '''Mark the pipelne as failed and record the given message.'''
        self.fail_time = self.add_message(message)
        self.status = PipelineStatus.FAILED

    def attributes(self):
        return [
            "name", "status", "start_time", "complete_time", "fail_time",
            "messages", "jobs",
        ]

    def write_json(self, path, pretty_print=True):
        '''Serializes the PipelineStatus object and writes it to disk. Then
        calls the instance's callback function, if any.

        Args:
            path: the output path
            pretty_print: when True (default), the resulting JSON will be
                outputted to be human readable; when False, it will be compact
                with no extra spaces or newline characters
        '''
        super(PipelineStatus, self).write_json(path, pretty_print=pretty_print)
        if self._callback:
            self._callback(path)  # invoke callback

    @classmethod
    def from_dict(cls, d):
        '''Constructs a PipelineStatus instance from a JSON dictionary.'''
        pipeline_status = PipelineStatus(d["name"])
        pipeline_status.status = d["status"]
        pipeline_status.start_time = d["start_time"]
        pipeline_status.complete_time = d["complete_time"]
        pipeline_status.fail_time = d["fail_time"]
        pipeline_status.messages = [
            StatusMessage.from_dict(_d) for _d in d["messages"]
        ]
        pipeline_status.jobs = [
            JobStatus.from_dict(_d) for _d in d["jobs"]
        ]
        return pipeline_status


class JobStatus(Serializable):
    '''Class for recording the status of a job.

    Attributes:
        name: the name of the job
        status: the current status of the job
        start_time: the time the job was started, or None if not started
        complete_time: the time the job was completed, or None if not completed
        fail_time: the time the job failed, or None if not failed
        messages: a list of StatusMessage objects listing the status updates
            for the job
    '''

    # Job status enum
    QUEUED = "QUEUED"
    SKIPPED = "SKIPPED"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"

    def __init__(self, name):
        '''Construct a new JobStatus instance.

        Args:
            name: the name of the job
        '''
        self.name = name
        self.status = JobStatus.QUEUED
        self.start_time = None
        self.complete_time = None
        self.fail_time = None
        self.messages = []

    def attributes(self):
        return [
            "name", "status", "start_time", "complete_time", "fail_time",
            "messages",
        ]

    def add_message(self, message):
        '''Add the given message to the messages list.'''
        status_message = StatusMessage(message)
        self.messages.append(status_message)
        return status_message.time

    def skip(self, message="Job skipped"):
        '''Mark the job as skipped and record the given message.'''
        self.add_message(message)
        self.status = JobStatus.SKIPPED

    def start(self, message="Job started"):
        '''Mark the job as started and record the given message.'''
        self.start_time = self.add_message(message)
        self.status = JobStatus.RUNNING

    def complete(self, message="Job completed"):
        '''Mark the job as complete and record the given message.'''
        self.complete_time = self.add_message(message)
        self.status = JobStatus.COMPLETE

    def fail(self, message="Job failed"):
        '''Mark the job as failed and record the given message.'''
        self.fail_time = self.add_message(message)
        self.status = JobStatus.FAILED

    @classmethod
    def from_dict(cls, d):
        '''Constructs a JobStatus instance from a JSON dictionary.'''
        job_status = JobStatus(d["name"])
        job_status.status = d["status"]
        job_status.start_time = d["start_time"]
        job_status.complete_time = d["complete_time"]
        job_status.fail_time = d["fail_time"]
        job_status.messages = [
            StatusMessage.from_dict(_d) for _d in d["messages"]
        ]
        return job_status


class StatusMessage(Serializable):
    '''Class encapsulating a status message with a timestamp.

    Attributes:
        message: the message string
        time: the string recording the message time
    '''

    def __init__(self, message, time=None):
        '''Creates a new StatusMessage instance.

        Args:
            message: a message string
            time: an optional time string. If not provided, the current time in
                 ISO 8601 format is used
        '''
        self.message = message
        self.time = time or etau.get_isotime()

    def attributes(self):
        return ["message", "time"]

    @classmethod
    def from_dict(cls, d):
        '''Constructs a StatusMessage instance from a JSON dictionary.'''
        return StatusMessage(d["message"], time=d["time"])
