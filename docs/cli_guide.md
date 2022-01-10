# Command-Line Interface Guide

Installing ETA automatically installs `eta`, a command-line interface (CLI) for
interacting with the ETA Library. This utility provides access to many useful
features of ETA, including building and running pipelines, downloading models,
and interacting with remote storage.

This document provides an overview of using the CLI.

## Quickstart

To see the available top-level commands, type `eta --help`.

You can learn more about any available subcommand via `eta <command> --help`.

For example, to see your current ETA config, you can execute `eta config`.

## Tab completion

To enable tab completion in `bash`, add the following line to your `~/.bashrc`:

```shell
eval "$(register-python-argcomplete eta)"
```

To enable tab completion in `zsh`, add these lines to your `~/.zshrc`:

```shell
autoload bashcompinit
bashcompinit
eval "$(register-python-argcomplete eta)"
```

To enable tab completion in `tcsh`, add these lines to your `~/.tcshrc`:

```shell
eval `register-python-argcomplete --shell tcsh eta`
```

## Usage

The following usage information was generated via `eta --all-help`:

> Last generated on 2020/06/18

```
*******************************************************************************
usage: eta [-h] [-v] [--all-help]
           {build,run,clean,models,modules,pipelines,constants,config,auth,s3,gcs,gdrive,http,sftp}
           ...

ETA command-line interface.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show version info
  --all-help            show help recurisvely and exit

available commands:
  {build,run,clean,models,modules,pipelines,constants,config,auth,s3,gcs,gdrive,http,sftp}
    build               Tools for building pipelines.
    run                 Tools for running pipelines and modules.
    clean               Tools for cleaning up after pipelines.
    models              Tools for working with models.
    modules             Tools for working with modules.
    pipelines           Tools for working with pipelines.
    constants           Print constants from `eta.constants`.
    config              Tools for working with your ETA config.
    auth                Tools for configuring authentication credentials.
    s3                  Tools for working with S3.
    gcs                 Tools for working with Google Cloud Storage.
    gdrive              Tools for working with Google Drive.
    http                Tools for working with HTTP storage.
    sftp                Tools for working with SFTP storage.


*******************************************************************************
usage: eta build [-h] [-n NAME] [-r REQUEST] [-i 'KEY=VAL,...']
                 [-o 'KEY=VAL,...'] [-p 'KEY=VAL,...'] [-e 'KEY=VAL,...']
                 [-l 'KEY=VAL,...'] [--patterns 'KEY=VAL,...'] [--unoptimized]
                 [--run-now] [--cleanup] [--debug]

Tools for building pipelines.

    Examples:
        # Build pipeline from a pipeline build request
        eta build -r '/path/to/pipeline/request.json'

        # Build pipeline request interactively, run it, and cleanup after
        eta build \
            -n video_formatter \
            -i 'video="examples/data/water.mp4"' \
            -o 'formatted_video="water-small.mp4"' \
            -p 'format_videos.scale=0.5' \
            --run-now --cleanup

optional arguments:
  -h, --help            show this help message and exit
  --unoptimized         don't optimize pipeline when building
  --run-now             run pipeline after building
  --cleanup             delete all generated files after running the pipeline
  --debug               set pipeline logging level to DEBUG

request arguments:
  -n NAME, --name NAME  pipeline name
  -r REQUEST, --request REQUEST
                        path to a PipelineBuildRequest file
  -i 'KEY=VAL,...', --inputs 'KEY=VAL,...'
                        pipeline inputs (can be repeated)
  -o 'KEY=VAL,...', --outputs 'KEY=VAL,...'
                        pipeline outputs (can be repeated)
  -p 'KEY=VAL,...', --parameters 'KEY=VAL,...'
                        pipeline parameters (can be repeated)
  -e 'KEY=VAL,...', --eta-config 'KEY=VAL,...'
                        ETA config settings (can be repeated)
  -l 'KEY=VAL,...', --logging 'KEY=VAL,...'
                        logging config settings (can be repeated)
  --patterns 'KEY=VAL,...'
                        patterns to replace in the build request (can be repeated)


*******************************************************************************
usage: eta run [-h] [-o] [-m MODULE] [-l] [config]

Tools for running pipelines and modules.

    Examples:
        # Run pipeline defined by a pipeline config
        eta run '/path/to/pipeline-config.json'

        # Run pipeline and force existing module outputs to be overwritten
        eta run --overwrite '/path/to/pipeline-config.json'

        # Run specified module with the given module config
        eta run --module <module-name> '/path/to/module-config.json'

        # Run last built pipeline
        eta run --last

positional arguments:
  config                path to PipelineConfig or ModuleConfig file

optional arguments:
  -h, --help            show this help message and exit
  -o, --overwrite       force overwrite existing module outputs
  -m MODULE, --module MODULE
                        run module with the given name
  -l, --last            run last built pipeline


*******************************************************************************
usage: eta clean [-h] [-l] [-a] [PATH]

Tools for cleaning up after pipelines.

    Examples:
        # Cleanup pipeline defined by a given pipeline config
        eta clean '/path/to/pipeline-config.json'

        # Cleanup last built pipeline
        eta clean --last

        # Cleanup all built pipelines
        eta clean --all

positional arguments:
  PATH        path to a PipelineConfig file

optional arguments:
  -h, --help  show this help message and exit
  -l, --last  cleanup the last built pipeline
  -a, --all   cleanup all built pipelines


*******************************************************************************
usage: eta models [-h] [-l] [--list-downloaded] [-s SEARCH] [-f NAME]
                  [-i NAME] [-d NAME] [--force-download NAME]
                  [--visualize-tf-graph NAME] [--init DIR] [--flush NAME]
                  [--flush-old] [--flush-all]

Tools for working with models.

    Examples:
        # List all available models
        eta models --list

        # List all downloaded models
        eta models --list-downloaded

        # Search for models whose names contains the given string
        eta models --search <search-str>

        # Find model
        eta models --find <model-name>

        # Print info about model
        eta models --info <model-name>

        # Download model, if necessary
        eta models --download <model-name>

        # Force download model
        eta models --force-download <model-name>

        # Visualize graph for model in TensorBoard (TF models only)
        eta models --visualize-tf-graph <model-name>

        # Initialize new models directory
        eta models --init <models-dir>

        # Flush given model
        eta models --flush <model-name>

        # Flush all old models
        eta models --flush-old

        # Flush all models
        eta models --flush-all

optional arguments:
  -h, --help            show this help message and exit
  -l, --list            list all published models on the current search path
  --list-downloaded     list all downloaded models on the current search path
  -s SEARCH, --search SEARCH
                        search for models whose names contain the given string
  -f NAME, --find NAME  find the model with the given name
  -i NAME, --info NAME  get info about the model with the given name
  -d NAME, --download NAME
                        download the model with the given name, if necessary
  --force-download NAME
                        force download the model with the given name
  --visualize-tf-graph NAME
                        visualize the TF graph for the model with the given name
  --init DIR            initialize the given models directory
  --flush NAME          flush the model with the given name
  --flush-old           flush all old models, i.e., those models for which the number of versions stored on disk exceeds `eta.config.max_model_versions_to_keep`
  --flush-all           flush all models


*******************************************************************************
usage: eta modules [-h] [-l] [-s SEARCH] [-f NAME] [-e NAME] [-i NAME]
                   [-d NAME] [-m PATH] [-r]

Tools for working with modules.

    Examples:
        # List all available modules
        eta modules --list

        # Search for modules whose names contain the given string
        eta modules --search <search-str>

        # Find metadata file for module
        eta modules --find <module-name>

        # Find executable file for module
        eta modules --find-exe <module-name>

        # Show metadata for module
        eta modules --info <module-name>

        # Generate block diagram for module
        eta modules --diagram <module-name>

        # Generate metadata file for module
        eta modules --metadata '/path/to/eta_module.py'

        # Refresh all module metadata files
        eta modules --refresh-metadata

optional arguments:
  -h, --help            show this help message and exit
  -l, --list            list all modules on search path
  -s SEARCH, --search SEARCH
                        search for modules whose names contain the given string
  -f NAME, --find NAME  find metadata file for module with the given name
  -e NAME, --find-exe NAME
                        find the module executable for module with the given name
  -i NAME, --info NAME  show metadata for module with the given name
  -d NAME, --diagram NAME
                        generate block diagram for module with the given name
  -m PATH, --metadata PATH
                        generate metadata file for the given module
  -r, --refresh-metadata
                        refresh all module metadata files


*******************************************************************************
usage: eta pipelines [-h] [-l] [-s NAME] [-f NAME] [-i NAME] [-d NAME]

Tools for working with pipelines.

    Examples:
        # List all available pipelines
        eta pipelines --list

        # Search for pipelines whose names contain the given string
        eta pipelines --search <search-str>

        # Find metadata file for pipeline
        eta pipelines --find <pipeline-name>

        # Show metadata for pipeline
        eta pipelines --info <pipeline-name>

        # Generate block diagram for pipeline
        eta pipelines --diagram <pipeline-name>

optional arguments:
  -h, --help            show this help message and exit
  -l, --list            list all ETA pipelines on the current search path
  -s NAME, --search NAME
                        search for pipelines whose names contain the given string
  -f NAME, --find NAME  find metadata file for pipeline with the given name
  -i NAME, --info NAME  show metadata for pipeline with the given name
  -d NAME, --diagram NAME
                        generate block diagram for pipeline with the given name


*******************************************************************************
usage: eta constants [-h] [CONSTANT]

Print constants from `eta.constants`.

    Examples:
        # Print all constants
        eta constants

        # Print a specific constant
        eta constants <CONSTANT>

positional arguments:
  CONSTANT    the constant to print

optional arguments:
  -h, --help  show this help message and exit


*******************************************************************************
usage: eta config [-h] [-l] [-s] [FIELD]

Tools for working with your ETA config.

    Examples:
        # Print your entire config
        eta config

        # Print a specific config field
        eta config <field>

        # Print the location of your config
        eta config --locate

        # Save your current config to disk
        eta config --save

positional arguments:
  FIELD         a config field

optional arguments:
  -h, --help    show this help message and exit
  -l, --locate  print the location of your ETA config on disk
  -s, --save    save your current config to disk


*******************************************************************************
usage: eta auth [-h] [--all-help] {show,activate,deactivate} ...

Tools for configuring authentication credentials.

optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recurisvely and exit

available commands:
  {show,activate,deactivate}
    show                Show info about active credentials.
    activate            Activate authentication credentials.
    deactivate          Deactivate authentication credentials.


*******************************************************************************
usage: eta auth show [-h] [--google] [--aws] [--ssh]

Show info about active credentials.

    Examples:
        # Print info about all active credentials
        eta auth show

        # Print info about active Google credentials
        eta auth show --google

        # Print info about active AWS credentials
        eta auth show --aws

        # Print info about active SSH credentials
        eta auth show --ssh

optional arguments:
  -h, --help  show this help message and exit
  --google    show info about Google credentials
  --aws       show info about AWS credentials
  --ssh       show info about SSH credentials


*******************************************************************************
usage: eta auth activate [-h] [--google PATH] [--aws PATH] [--ssh PATH]

Activate authentication credentials.

    Examples:
        # Activate Google credentials
        eta auth activate --google '/path/to/service-account.json'

        # Activate AWS credentials
        eta auth activate --aws '/path/to/credentials.ini'

        # Activate SSH credentials
        eta auth activate --ssh '/path/to/id_rsa'

optional arguments:
  -h, --help     show this help message and exit
  --google PATH  path to Google service account JSON file
  --aws PATH     path to AWS credentials file
  --ssh PATH     path to SSH private key


*******************************************************************************
usage: eta auth deactivate [-h] [--google] [--aws] [--ssh] [--all]

Deactivate authentication credentials.

    Examples:
        # Deactivate Google credentials
        eta auth deactivate --google

        # Deactivate AWS credentials
        eta auth deactivate --aws

        # Deactivate SSH credentials
        eta auth deactivate --ssh

        # Deactivate all credentials
        eta auth deactivate --all

optional arguments:
  -h, --help  show this help message and exit
  --google    delete the active Google credentials
  --aws       delete the active AWS credentials
  --ssh       delete the active SSH credentials
  --all       delete all active credentials


*******************************************************************************
usage: eta s3 [-h] [--all-help]
              {info,list,upload,upload-dir,download,download-dir,delete,delete-dir}
              ...

Tools for working with S3.

optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recurisvely and exit

available commands:
  {info,list,upload,upload-dir,download,download-dir,delete,delete-dir}
    info                Get information about files/folders in S3.
    list                List contents of an S3 folder.
    upload              Upload file to S3.
    upload-dir          Upload directory to S3.
    download            Download file from S3.
    download-dir        Download directory from S3.
    delete              Delete file from S3.
    delete-dir          Delete directory from S3.


*******************************************************************************
usage: eta s3 info [-h] [-f] CLOUD_PATH [CLOUD_PATH ...]

Get information about files/folders in S3.

    Examples:
        # Get file info
        eta s3 info <cloud-path> [...]

        # Get folder info
        eta s3 info --folder <cloud-path> [...]

positional arguments:
  CLOUD_PATH    the path(s) of the files of interest in S3

optional arguments:
  -h, --help    show this help message and exit
  -f, --folder  whether the providedpaths are folders, not files


*******************************************************************************
usage: eta s3 list [-h] [-r] [-l LIMIT] [-s SEARCH] [--sort-by FIELD]
                   [--ascending] [-c]
                   CLOUD_DIR

List contents of an S3 folder.

    Examples:
        # List folder contents
        eta s3 list s3://<bucket>/<prefix>

        # List folder contents recursively
        eta s3 list s3://<bucket>/<prefix> --recursive

        # List folder contents according to the given query
        eta s3 list s3://<bucket>/<prefix>
            [--recursive]
            [--limit <limit>]
            [--search [<field><operator>]<str>[,...]]
            [--sort-by <field>]
            [--ascending]
            [--count]

        # List the last 10 modified files that contain "test" in any field
        eta s3 list s3://<bucket>/<prefix> \
            --search test --limit 10 --sort-by last_modified

        # List files whose size is 10-20MB, from smallest to largest
        eta s3 list s3://<bucket>/<prefix> \
            --search 'size>10MB,size<20MB' --sort-by size --ascending

        # List files that were uploaded before November 26th, 2019, recurisvely
        # traversing subfolders, and display the count
        eta s3 list s3://<bucket>/<prefix> \
            --recursive --search 'last modified<2019-11-26' --count

    Search syntax:
        The generic search syntax is:

            --search [<field><operator>]<str>[,...]

        where:
            <field>    an optional field name on which to search
            <operator> an optional operator to use when evaluating matches
            <str>      the search string

        If <field><operator> is omitted, the search will match any records for
        which any column contains the given search string.

        Multiple searches can be specified as a comma-separated list. Records
        must match all searches in order to appear in the search results.

        The supported fields are:

        field         type     description
        ------------- -------- ------------------------------------------
        bucket        string   the name of the bucket
        name          string   the name of the object in the bucket
        size          bytes    the size of the object
        type          string   the MIME type of the object
        last modified datetime the date that the object was last modified

        Fields are case insensitive, and underscores can be used in-place of
        spaces.

        The meaning of the operators are as follows:

        operator  type       description
        --------- ---------- --------------------------------------------------
        :         contains   the field contains the search string
        ==        comparison the search string is equal to the field
        <         comparison the search string is less than the field
        <=        comparison the search string is less or equal to the field
        >         comparison the search string is greater than the field
        >=        comparison the search string is greater or equal to the field

        For contains (":") queries, the search/record values are parsed as
        follows:

        type     description
        -------- --------------------------------------------------------------
        string   the search and record are treated as strings
        bytes    the search is treated as a string, and the record is converted
                 to a human-readable bytes string
        datetime the search is treated as a string, and the record is rendered
                 as a string in "%Y-%m-%d %H:%M:%S %Z" format in local timezone

        For comparison ("==", "<", "<=", ">", ">=") queries, the search/record
        values are parsed as follows:

        type     description
        -------- ------------------------------------------------------------
        string   the search and record are treated as strings
        bytes    the search must be a human-readable bytes string, which is
                 converted to numeric bytes for comparison with the record
        datetime the search must be an ISO time string, which is converted to
                 a datetime for comparison with the record. If no timezone is
                 included in the search, local time is assumed

        You can include special characters (":", "=", "<", ">", ",") in search
        strings by escaping them with "\".

positional arguments:
  CLOUD_DIR             the S3 folder to list

optional arguments:
  -h, --help            show this help message and exit
  -r, --recursive       whether to recursively list the contents of subfolders
  -l LIMIT, --limit LIMIT
                        limit the number of files listed
  -s SEARCH, --search SEARCH
                        search to limit results when listing files
  --sort-by FIELD       field to sort by when listing files
  --ascending           whether to sort in ascending order
  -c, --count           whether to show the number of files in the list


*******************************************************************************
usage: eta s3 upload [-h] [-t TYPE] LOCAL_PATH CLOUD_PATH

Upload file to S3.

    Examples:
        # Upload file
        eta s3 upload <local-path> <cloud-path>

positional arguments:
  LOCAL_PATH            the path to the file to upload
  CLOUD_PATH            the path to the S3 object to create

optional arguments:
  -h, --help            show this help message and exit
  -t TYPE, --content-type TYPE
                        an optional content type of the file. By default, the type is guessed from the filename


*******************************************************************************
usage: eta s3 upload-dir [-h] [--sync] [-o] [-r] LOCAL_DIR CLOUD_DIR

Upload directory to S3.

    Examples:
        # Upload directory
        eta s3 upload-dir <local-dir> <cloud-dir>

        # Upload-sync directory
        eta s3 upload-dir --sync <local-dir> <cloud-dir>

positional arguments:
  LOCAL_DIR        the directory of files to upload
  CLOUD_DIR        the S3 directory to upload into

optional arguments:
  -h, --help       show this help message and exit
  --sync           whether to sync the S3 directory to match the contents of the local directory
  -o, --overwrite  whether to overwrite existing files; only valid in `--sync` mode
  -r, --recursive  whether to recursively upload the contents of subdirecotires


*******************************************************************************
usage: eta s3 download [-h] [--print] CLOUD_PATH [LOCAL_PATH]

Download file from S3.

    Examples:
        # Download file
        eta s3 download <cloud-path> <local-path>

        # Print download to stdout
        eta s3 download <cloud-path> --print

positional arguments:
  CLOUD_PATH  the S3 object to download
  LOCAL_PATH  the path to which to write the downloaded file. If not provided, the filename of the file in S3 is used

optional arguments:
  -h, --help  show this help message and exit
  --print     whether to print the download to stdout. If true, a file is NOT written to disk


*******************************************************************************
usage: eta s3 download-dir [-h] [--sync] [-o] [-r] CLOUD_DIR LOCAL_DIR

Download directory from S3.

    Examples:
        # Download directory
        eta s3 download-dir <cloud-folder> <local-dir>

        # Download directory sync
        eta s3 download-dir --sync <cloud-folder> <local-dir>

positional arguments:
  CLOUD_DIR        the S3 directory to download
  LOCAL_DIR        the directory to which to download files into

optional arguments:
  -h, --help       show this help message and exit
  --sync           whether to sync the localdirectory to match the contents of the S3 directory
  -o, --overwrite  whether to overwrite existing files; only valid in `--sync` mode
  -r, --recursive  whether to recursively download the contents of subdirecotires


*******************************************************************************
usage: eta s3 delete [-h] CLOUD_PATH

Delete file from S3.

    Examples:
        # Delete file
        eta s3 delete <cloud-path>

positional arguments:
  CLOUD_PATH  the S3 file to delete

optional arguments:
  -h, --help  show this help message and exit


*******************************************************************************
usage: eta s3 delete-dir [-h] CLOUD_DIR

Delete directory from S3.

    Examples:
        # Delete directory
        eta s3 delete-dir <cloud-dir>

positional arguments:
  CLOUD_DIR   the S3 folder to delete

optional arguments:
  -h, --help  show this help message and exit


*******************************************************************************
usage: eta gcs [-h] [--all-help]
               {info,list,upload,upload-dir,download,download-dir,delete,delete-dir}
               ...

Tools for working with Google Cloud Storage.

optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recurisvely and exit

available commands:
  {info,list,upload,upload-dir,download,download-dir,delete,delete-dir}
    info                Get information about files/folders in GCS.
    list                List contents of a GCS folder.
    upload              Upload file to GCS.
    upload-dir          Upload directory to GCS.
    download            Download file from GCS.
    download-dir        Download directory from GCS.
    delete              Delete file from GCS.
    delete-dir          Delete directory from GCS.


*******************************************************************************
usage: eta gcs info [-h] [-f] CLOUD_PATH [CLOUD_PATH ...]

Get information about files/folders in GCS.

    Examples:
        # Get file info
        eta gcs info <cloud-path> [...]

        # Get folder info
        eta gcs info --folder <cloud-path> [...]

positional arguments:
  CLOUD_PATH    path(s) to GCS files

optional arguments:
  -h, --help    show this help message and exit
  -f, --folder  whether the providedpaths are folders, not files


*******************************************************************************
usage: eta gcs list [-h] [-r] [-l LIMIT] [-s SEARCH] [--sort-by FIELD]
                    [--ascending] [-c]
                    CLOUD_DIR

List contents of a GCS folder.

    Examples:
        # List folder contents
        eta gcs list gs://<bucket>/<prefix>

        # List folder contents recursively
        eta gcs list gs://<bucket>/<prefix> --recursive

        # List folder contents according to the given query
        eta gcs list gs://<bucket>/<prefix>
            [--recursive]
            [--limit <limit>]
            [--search [<field><operator>]<str>[,...]]
            [--sort-by <field>]
            [--ascending]
            [--count]

        # List the last 10 modified files that contain "test" in any field
        eta gcs list gs://<bucket>/<prefix> \
            --search test --limit 10 --sort-by last_modified

        # List files whose size is 10-20MB, from smallest to largest
        eta gcs list gs://<bucket>/<prefix> \
            --search 'size>10MB,size<20MB' --sort-by size --ascending

        # List files that were uploaded before November 26th, 2019, recurisvely
        # traversing subfolders, and display the count
        eta gcs list gs://<bucket>/<prefix> \
            --recursive --search 'last modified<2019-11-26' --count

    Search syntax:
        The generic search syntax is:

            --search [<field><operator>]<str>[,...]

        where:
            <field>    an optional field name on which to search
            <operator> an optional operator to use when evaluating matches
            <str>      the search string

        If <field><operator> is omitted, the search will match any records for
        which any column contains the given search string.

        Multiple searches can be specified as a comma-separated list. Records
        must match all searches in order to appear in the search results.

        The supported fields are:

        field         type     description
        ------------- -------- ------------------------------------------
        bucket        string   the name of the bucket
        name          string   the name of the object in the bucket
        size          bytes    the size of the object
        type          string   the MIME type of the object
        last modified datetime the date that the object was last modified

        Fields are case insensitive, and underscores can be used in-place of
        spaces.

        The meaning of the operators are as follows:

        operator  type       description
        --------- ---------- --------------------------------------------------
        :         contains   the field contains the search string
        ==        comparison the search string is equal to the field
        <         comparison the search string is less than the field
        <=        comparison the search string is less or equal to the field
        >         comparison the search string is greater than the field
        >=        comparison the search string is greater or equal to the field

        For contains (":") queries, the search/record values are parsed as
        follows:

        type     description
        -------- --------------------------------------------------------------
        string   the search and record are treated as strings
        bytes    the search is treated as a string, and the record is converted
                 to a human-readable bytes string
        datetime the search is treated as a string, and the record is rendered
                 as a string in "%Y-%m-%d %H:%M:%S %Z" format in local timezone

        For comparison ("==", "<", "<=", ">", ">=") queries, the search/record
        values are parsed as follows:

        type     description
        -------- ------------------------------------------------------------
        string   the search and record are treated as strings
        bytes    the search must be a human-readable bytes string, which is
                 converted to numeric bytes for comparison with the record
        datetime the search must be an ISO time string, which is converted to
                 a datetime for comparison with the record. If no timezone is
                 included in the search, local time is assumed

        You can include special characters (":", "=", "<", ">", ",") in search
        strings by escaping them with "\".

positional arguments:
  CLOUD_DIR             the GCS folder to list

optional arguments:
  -h, --help            show this help message and exit
  -r, --recursive       whether to recursively list the contents of subfolders
  -l LIMIT, --limit LIMIT
                        limit the number of files listed
  -s SEARCH, --search SEARCH
                        search to limit results when listing files
  --sort-by FIELD       field to sort by when listing files
  --ascending           whether to sort in ascending order
  -c, --count           whether to show the number of files in the list


*******************************************************************************
usage: eta gcs upload [-h] [-t TYPE] [-s SIZE] LOCAL_PATH CLOUD_PATH

Upload file to GCS.

    Examples:
        # Upload file
        eta gcs upload <local-path> <cloud-path>

positional arguments:
  LOCAL_PATH            the path to the file to upload
  CLOUD_PATH            the path to the GCS object to create

optional arguments:
  -h, --help            show this help message and exit
  -t TYPE, --content-type TYPE
                        an optional content type of the file. By default, the type is guessed from the filename
  -s SIZE, --chunk-size SIZE
                        an optional chunk size (in bytes) to use


*******************************************************************************
usage: eta gcs upload-dir [-h] [--sync] [-o] [-r] [-s SIZE]
                          LOCAL_DIR CLOUD_DIR

Upload directory to GCS.

    Examples:
        # Upload directory
        eta gcs upload-dir <local-dir> <cloud-dir>

        # Upload-sync directory
        eta gcs upload-dir --sync <local-dir> <cloud-dir>

positional arguments:
  LOCAL_DIR             the directory of files to upload
  CLOUD_DIR             the GCS directory to upload into

optional arguments:
  -h, --help            show this help message and exit
  --sync                whether to sync the GCSdirectory to match the contents of the local directory
  -o, --overwrite       whether to overwrite existing files; only valid in `--sync` mode
  -r, --recursive       whether to recursively upload the contents of subdirecotires
  -s SIZE, --chunk-size SIZE
                        an optional chunk size (in bytes) to use


*******************************************************************************
usage: eta gcs download [-h] [--print] [-s SIZE] CLOUD_PATH [LOCAL_PATH]

Download file from GCS.

    Examples:
        # Download file
        eta gcs download <cloud-path> <local-path>

        # Print download to stdout
        eta gcs download <cloud-path> --print

positional arguments:
  CLOUD_PATH            the GCS object to download
  LOCAL_PATH            the path to which to write the downloaded file. If not provided, the filename of the file in GCS is used

optional arguments:
  -h, --help            show this help message and exit
  --print               whether to print the download to stdout. If true, a file is NOT written to disk
  -s SIZE, --chunk-size SIZE
                        an optional chunk size (in bytes) to use


*******************************************************************************
usage: eta gcs download-dir [-h] [--sync] [-o] [-r] [-s SIZE]
                            CLOUD_DIR LOCAL_DIR

Download directory from GCS.

    Examples:
        # Download directory
        eta gcs download-dir <cloud-folder> <local-dir>

        # Download directory sync
        eta gcs download-dir --sync <cloud-folder> <local-dir>

positional arguments:
  CLOUD_DIR             the GCS directory to download
  LOCAL_DIR             the directory to which to download files into

optional arguments:
  -h, --help            show this help message and exit
  --sync                whether to sync the localdirectory to match the contents of the GCS directory
  -o, --overwrite       whether to overwrite existing files; only valid in `--sync` mode
  -r, --recursive       whether to recursively download the contents of subdirecotires
  -s SIZE, --chunk-size SIZE
                        an optional chunk size (in bytes) to use


*******************************************************************************
usage: eta gcs delete [-h] CLOUD_PATH

Delete file from GCS.

    Examples:
        # Delete file
        eta gcs delete <cloud-path>

positional arguments:
  CLOUD_PATH  the GCS file to delete

optional arguments:
  -h, --help  show this help message and exit


*******************************************************************************
usage: eta gcs delete-dir [-h] CLOUD_DIR

Delete directory from GCS.

    Examples:
        # Delete directory
        eta gcs delete-dir <cloud-dir>

positional arguments:
  CLOUD_DIR   the GCS directory to delete

optional arguments:
  -h, --help  show this help message and exit


*******************************************************************************
usage: eta gdrive [-h] [--all-help]
                  {info,list,upload,upload-dir,download,download-dir,delete,delete-dir}
                  ...

Tools for working with Google Drive.

optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recurisvely and exit

available commands:
  {info,list,upload,upload-dir,download,download-dir,delete,delete-dir}
    info                Get information about files/folders in Google Drive.
    list                List contents of a Google Drive folder.
    upload              Upload file to Google Drive.
    upload-dir          Upload directory to Google Drive.
    download            Download file from Google Drive.
    download-dir        Download directory from Google Drive.
    delete              Delete file from Google Drive.
    delete-dir          Delete directory from Google Drive.


*******************************************************************************
usage: eta gdrive info [-h] [-f] ID [ID ...]

Get information about files/folders in Google Drive.

    Examples:
        # Get file info
        eta gdrive info <file-id> [...]

        # Get folder info
        eta gdrive info --folder <folder-id> [...]

positional arguments:
  ID            the ID(s) of the files of interest in Google Drive

optional arguments:
  -h, --help    show this help message and exit
  -f, --folder  whether the files ofinterest are folders


*******************************************************************************
usage: eta gdrive list [-h] [-r] [-l LIMIT] [-s SEARCH] [--sort-by FIELD]
                       [--ascending] [-c]
                       ID

List contents of a Google Drive folder.

    Examples:
        # List folder contents
        eta gdrive list <folder-id>

        # List folder contents recursively
        eta gdrive list <folder-id> --recursive

        # List folder contents according to the given query
        eta gdrive list <folder-id>
            [--recursive]
            [--limit <limit>]
            [--search [<field><operator>]<str>[,...]]
            [--sort-by <field>]
            [--ascending]
            [--count]

        # List the last 10 modified files that contain "test" in any field
        eta gdrive list <folder-id> \
            --search test --limit 10 --sort-by last_modified

        # List files whose size is 10-20MB, from smallest to largest
        eta gdrive list <folder-id> \
            --search 'size>10MB,size<20MB' --sort-by size --ascending

        # List files that were uploaded before November 26th, 2019, recurisvely
        # traversing subfolders, and display the count
        eta gdrive list <folder-id> \
            --recursive --search 'last modified<2019-11-26' --count

    Search syntax:
        The generic search syntax is:

            --search [<field><operator>]<str>[,...]

        where:
            <field>    an optional field name on which to search
            <operator> an optional operator to use when evaluating matches
            <str>      the search string

        If <field><operator> is omitted, the search will match any records for
        which any column contains the given search string.

        Multiple searches can be specified as a comma-separated list. Records
        must match all searches in order to appear in the search results.

        The supported fields are:

        field         type     description
        ------------- -------- ------------------------------------------
        id            string   the ID of the file
        name          string   the name of the file
        size          bytes    the size of the file
        type          string   the MIME type of the object
        last modified datetime the date that the file was last modified

        Fields are case insensitive, and underscores can be used in-place of
        spaces.

        The meaning of the operators are as follows:

        operator  type       description
        --------- ---------- --------------------------------------------------
        :         contains   the field contains the search string
        ==        comparison the search string is equal to the field
        <         comparison the search string is less than the field
        <=        comparison the search string is less or equal to the field
        >         comparison the search string is greater than the field
        >=        comparison the search string is greater or equal to the field

        For contains (":") queries, the search/record values are parsed as
        follows:

        type     description
        -------- --------------------------------------------------------------
        string   the search and record are treated as strings
        bytes    the search is treated as a string, and the record is converted
                 to a human-readable bytes string
        datetime the search is treated as a string, and the record is rendered
                 as a string in "%Y-%m-%d %H:%M:%S %Z" format in local timezone

        For comparison ("==", "<", "<=", ">", ">=") queries, the search/record
        values are parsed as follows:

        type     description
        -------- ------------------------------------------------------------
        string   the search and record are treated as strings
        bytes    the search must be a human-readable bytes string, which is
                 converted to numeric bytes for comparison with the record
        datetime the search must be an ISO time string, which is converted to
                 a datetime for comparison with the record. If no timezone is
                 included in the search, local time is assumed

        You can include special characters (":", "=", "<", ">", ",") in search
        strings by escaping them with "\".

positional arguments:
  ID                    the ID of the folder to list

optional arguments:
  -h, --help            show this help message and exit
  -r, --recursive       whether to recursively list the contents of subfolders
  -l LIMIT, --limit LIMIT
                        limit the number of files listed
  -s SEARCH, --search SEARCH
                        search to limit results when listing files
  --sort-by FIELD       field to sort by when listing files
  --ascending           whether to sort in ascending order
  -c, --count           whether to show the number of files in the list


*******************************************************************************
usage: eta gdrive upload [-h] [-f FILENAME] [-t TYPE] [-s SIZE] LOCAL_PATH ID

Upload file to Google Drive.

    Examples:
        # Upload file
        eta gdrive upload <local-path> <folder-id>

positional arguments:
  LOCAL_PATH            the path to the file to upload
  ID                    the ID of the folder to upload the file into

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        an optional filename to include in the request. By default, the name of the local file is used
  -t TYPE, --content-type TYPE
                        an optional content type of the file. By default, the type is guessed from the filename
  -s SIZE, --chunk-size SIZE
                        an optional chunk size (in bytes) to use


*******************************************************************************
usage: eta gdrive upload-dir [-h] [-f] [-e] [-r] [-s SIZE] LOCAL_DIR ID

Upload directory to Google Drive.

    Examples:
        # Upload directory
        eta gdrive upload-dir <local-dir> <folder-id>

positional arguments:
  LOCAL_DIR             the directory of files to upload
  ID                    the ID of the folder to upload the files into

optional arguments:
  -h, --help            show this help message and exit
  -f, --skip-failures   whether to skip failures
  -e, --skip-existing   whether to skip existing files
  -r, --recursive       whether to recursively upload the contents of subdirecotires
  -s SIZE, --chunk-size SIZE
                        an optional chunk size (in bytes) to use


*******************************************************************************
usage: eta gdrive download [-h] [--public] [--print] [-s SIZE] ID [LOCAL_PATH]

Download file from Google Drive.

    Examples:
        # Download file
        eta gdrive download <file-id> <local-path>

        # Print download to stdout
        eta gdrive download <file-id> --print

        # Download file with link sharing turned on (no credentials required)
        eta gdrive download --public <file-id> <local-path>

positional arguments:
  ID                    the ID of the file to download
  LOCAL_PATH            the path to which to write the downloaded file. If not provided, the filename of the file in Google Drive is used

optional arguments:
  -h, --help            show this help message and exit
  --public              whether the file has public link sharing turned on and can therefore be downloaded with no credentials
  --print               whether to print the download to stdout. If true, a file is NOT written to disk
  -s SIZE, --chunk-size SIZE
                        an optional chunk size (in bytes) to use


*******************************************************************************
usage: eta gdrive download-dir [-h] [-f] [-e] [-r] [-s SIZE] ID LOCAL_DIR

Download directory from Google Drive.

    Examples:
        # Download directory
        eta gdrive download-dir <folder-id> <local-dir>

positional arguments:
  ID                    the ID of the folder to download
  LOCAL_DIR             the directory to download the files into

optional arguments:
  -h, --help            show this help message and exit
  -f, --skip-failures   whether to skip failures
  -e, --skip-existing   whether to skip existing files
  -r, --recursive       whether to recursively download the contents of subdirecotires
  -s SIZE, --chunk-size SIZE
                        an optional chunk size (in bytes) to use


*******************************************************************************
usage: eta gdrive delete [-h] ID

Delete file from Google Drive.

    Examples:
        # Delete file
        eta gdrive delete <file-id>

positional arguments:
  ID          the ID of the file to delete

optional arguments:
  -h, --help  show this help message and exit


*******************************************************************************
usage: eta gdrive delete-dir [-h] [-c] [-s] ID

Delete directory from Google Drive.

    Examples:
        # Delete directory
        eta gdrive delete-dir <folder-id>

        # Delete the contents (only) of a directory
        eta gdrive delete-dir <folder-id> --contents-only

positional arguments:
  ID                   the ID of the folder to delete

optional arguments:
  -h, --help           show this help message and exit
  -c, --contents-only  whether to delete only the contents of the folder (not the folder itself)
  -s, --skip-failures  whether to skip failures


*******************************************************************************
usage: eta http [-h] [--all-help] {upload,download,delete} ...

Tools for working with HTTP storage.

optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recurisvely and exit

available commands:
  {upload,download,delete}
    upload              Upload file via HTTP.
    download            Download file via HTTP.
    delete              Delete file via HTTP.


*******************************************************************************
usage: eta http upload [-h] [-f FILENAME] [-t TYPE] LOCAL_PATH URL

Upload file via HTTP.

    Examples:
        # Upload file
        eta http upload <local-path> <url>

positional arguments:
  LOCAL_PATH            the path to the file to upload
  URL                   the URL to which to PUT the file

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        an optional filename to include in the request. By default, the name of the local file is used
  -t TYPE, --content-type TYPE
                        an optional content type of the file. By default, the type is guessed from the filename


*******************************************************************************
usage: eta http download [-h] [--print] [-s SIZE] URL [LOCAL_PATH]

Download file via HTTP.

    Examples:
        # Download file
        eta http download <url> <local-path>

        # Print download to stdout
        eta http download <url> --print

positional arguments:
  URL                   the URL from which to GET the file
  LOCAL_PATH            the path to which to write the downloaded file. If not provided, the filename is guessed from the URL

optional arguments:
  -h, --help            show this help message and exit
  --print               whether to print the download to stdout. If true, a file is NOT written to disk
  -s SIZE, --chunk-size SIZE
                        an optional chunk size (in bytes) to use


*******************************************************************************
usage: eta http delete [-h] URL

Delete file via HTTP.

    Examples:
        # Delete file
        eta http delete <url>

positional arguments:
  URL         the URL to DELETE

optional arguments:
  -h, --help  show this help message and exit


*******************************************************************************
usage: eta sftp [-h] [--all-help]
                {upload,upload-dir,download,download-dir,delete,delete-dir}
                ...

Tools for working with SFTP storage.

optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recurisvely and exit

available commands:
  {upload,upload-dir,download,download-dir,delete,delete-dir}
    upload              Upload file via SFTP.
    upload-dir          Upload directory via SFTP.
    download            Download file via SFTP.
    download-dir        Download directory via SFTP.
    delete              Delete file via SFTP.
    delete-dir          Delete directory via SFTP.


*******************************************************************************
usage: eta sftp upload [-h] [--user USER] [--host HOST] [-p PORT]
                       LOCAL_PATH REMOTE_PATH

Upload file via SFTP.

    Examples:
        # Upload file
        eta sftp upload <local-path> <user>@<host>:<remote-path>
        eta sftp upload --user <user> --host <host> <local-path> <remote-path>

positional arguments:
  LOCAL_PATH            the path to the file to upload
  REMOTE_PATH           the remote path to write the file

optional arguments:
  -h, --help            show this help message and exit
  --user USER           the username
  --host HOST           the hostname
  -p PORT, --port PORT  the port to use


*******************************************************************************
usage: eta sftp upload-dir [-h] [--user USER] [--host HOST] [-p PORT]
                           LOCAL_DIR REMOTE_DIR

Upload directory via SFTP.

    Examples:
        # Upload directory
        eta sftp upload-dir <local-dir> <user>@<host>:<remote-dir>
        eta sftp upload-dir --user <user> --host <host> <local-dir> <remote-dir>

positional arguments:
  LOCAL_DIR             the path to the directory to upload
  REMOTE_DIR            the remote directory to write the uploaded directory

optional arguments:
  -h, --help            show this help message and exit
  --user USER           the username
  --host HOST           the hostname
  -p PORT, --port PORT  the port to use


*******************************************************************************
usage: eta sftp download [-h] [--user USER] [--host HOST] [-p PORT] [--print]
                         REMOTE_PATH [LOCAL_PATH]

Download file via SFTP.

    Examples:
        # Download file
        eta sftp download <user>@<host>:<remote-path> <local-path>
        eta sftp download --user <user> --host <host> <remote-path> <local-path>

        # Print download to stdout
        eta sftp download <remote-path> --print

positional arguments:
  REMOTE_PATH           the remote file to download
  LOCAL_PATH            the path to which to write the downloaded file. If not provided, the filename is guessed from the remote path

optional arguments:
  -h, --help            show this help message and exit
  --user USER           the username
  --host HOST           the hostname
  -p PORT, --port PORT  the port to use
  --print               whether to print the download to stdout. If true, a file is NOT written to disk


*******************************************************************************
usage: eta sftp download-dir [-h] [--user USER] [--host HOST] [-p PORT]
                             REMOTE_DIR LOCAL_DIR

Download directory via SFTP.

    Examples:
        # Download directory
        eta sftp download-dir <user>@<host>:<remote-dir> <local-dir>
        eta sftp download-dir --user <user> --host <host> <remote-dir> <local-dir>

positional arguments:
  REMOTE_DIR            the remote directory to download
  LOCAL_DIR             the local directory to write the downloaded directory

optional arguments:
  -h, --help            show this help message and exit
  --user USER           the username
  --host HOST           the hostname
  -p PORT, --port PORT  the port to use


*******************************************************************************
usage: eta sftp delete [-h] [--user USER] [--host HOST] [-p PORT] REMOTE_PATH

Delete file via SFTP.

    Examples:
        # Delete file
        eta sftp delete <user>@<host>:<remote-path>
        eta sftp delete --user <user> --host <host> <remote-path>

positional arguments:
  REMOTE_PATH           the remote file to delete

optional arguments:
  -h, --help            show this help message and exit
  --user USER           the username
  --host HOST           the hostname
  -p PORT, --port PORT  the port to use


*******************************************************************************
usage: eta sftp delete-dir [-h] [--user USER] [--host HOST] [-p PORT]
                           REMOTE_DIR

Delete directory via SFTP.

    Examples:
        # Delete directory
        eta sftp delete <user>@<host>:<remote-dir>
        eta sftp delete --user <user> --host <host> <remote-dir>

positional arguments:
  REMOTE_DIR            the remote directory to delete

optional arguments:
  -h, --help            show this help message and exit
  --user USER           the username
  --host HOST           the hostname
  -p PORT, --port PORT  the port to use
```

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
