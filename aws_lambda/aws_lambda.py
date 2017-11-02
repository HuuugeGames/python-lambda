# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import json
import logging
import os
import sys
import time
from collections import defaultdict
from imp import load_source
from shutil import copy
from shutil import copyfile
from shutil import copytree
from tempfile import mkdtemp

import boto3
import botocore
import pip
import yaml
from pip._vendor.distlib._backport import shutil

from .helpers import archive
from .helpers import get_environment_variable_value
from .helpers import mkdir
from .helpers import read

ARN_PREFIXES = {
    'us-gov-west-1': 'aws-us-gov',
}

log = logging.getLogger(__name__)


def cleanup_old_versions(src, keep_last_versions, config_file_path=None, aws_profile=None):
    """Deletes old deployed versions of the function in AWS Lambda.

    Won't delete $Latest and any aliased version

    :param config_file_path: path to custom config.yaml file
    :param str src:
        The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param int keep_last_versions:
        The number of recent versions to keep and not delete
    """
    if keep_last_versions <= 0:
        print("Won't delete all versions. Please do this manually")
    else:
        cfg = read_config_file(config_file_path, src)

        client = get_client('lambda', cfg, aws_profile=aws_profile)

        response = client.list_versions_by_function(
            FunctionName=cfg.get('function_name'),
        )
        versions = response.get('Versions')
        if len(response.get('Versions')) < keep_last_versions:
            print('Nothing to delete. (Too few versions published)')
        else:
            version_numbers = [elem.get('Version') for elem in
                               versions[1:-keep_last_versions]]
            for version_number in version_numbers:
                try:
                    client.delete_function(
                        FunctionName=cfg.get('function_name'),
                        Qualifier=version_number,
                    )
                except botocore.exceptions.ClientError as e:
                    print('Skipping Version {}: {}'
                          .format(version_number, e.message))


def deploy(src, requirements=False, local_package=None, upload_to_s3=False, config_file_path=None, aws_profile=None):
    """Deploys a new function to AWS Lambda.

    :param aws_profile: aws profile name stored in ~/.aws/credentials
    :param config_file_path: path to custom config file
    :param upload_to_s3: upload function code to S3
    :param requirements:
    :param str src:
        The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param str local_package:
        The path to a local package with should be included in the deploy as
        well (and/or is not available on PyPi)
    """
    # Load and parse the config file.
    cfg = read_config_file(config_file_path, src)

    # Copy all the pip dependencies required to run your code into a temporary
    # folder then add the handler file in the root of this directory.
    # Zip the contents of this folder into a single file and output to the dist
    # directory.
    path_to_zip_file = build(src, requirements, local_package, config_file_path=config_file_path)
    filename = upload_s3(cfg, path_to_zip_file, aws_profile=aws_profile) if upload_to_s3 else None

    if function_exists(cfg, cfg.get('function_name'), aws_profile=aws_profile):
        update_function(cfg, path_to_zip_file, upload_to_s3, filename=filename, aws_profile=aws_profile)
    else:
        create_function(cfg, path_to_zip_file, upload_to_s3, filename=filename, aws_profile=aws_profile)
    if cfg.get('trigger'):
        create_trigger(cfg, aws_profile=aws_profile)


def upload(src, requirements=False, local_package=None, config_file_path=None, aws_profile=None):
    """Uploads a new function to AWS S3.

    :param aws_profile: aws profile name stored in ~/.aws/credentials
    :param config_file_path: path to custom config.yaml file
    :param requirements:
    :param str src:
        The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param str local_package:
        The path to a local package with should be included in the deploy as
        well (and/or is not available on PyPi)
    """
    # Load and parse the config file.
    cfg = read_config_file(config_file_path, src)

    # Copy all the pip dependencies required to run your code into a temporary
    # folder then add the handler file in the root of this directory.
    # Zip the contents of this folder into a single file and output to the dist
    # directory.
    path_to_zip_file = build(src, requirements, local_package)

    upload_s3(cfg, path_to_zip_file, aws_profile=aws_profile)


def invoke(src, alt_event=None, verbose=False, config_file_path=None):
    """Simulates a call to your function.

    :param config_file_path: path to custom config.yaml file
    :param str src:
        The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param str alt_event:
        An optional argument to override which event file to use.
    :param bool verbose:
        Whether to print out verbose details.
    """
    # Load and parse the config file.
    cfg = read_config_file(config_file_path, src)

    # Load environment variables from the config file into the actual
    # environment.
    env_vars = cfg.get('environment_variables')
    if env_vars:
        for key, value in env_vars.items():
            os.environ[key] = value

    # Load and parse event file.
    if alt_event:
        path_to_event_file = os.path.join(src, alt_event)
    else:
        path_to_event_file = os.path.join(src, 'event.json')
    event = read(path_to_event_file, loader=json.loads)

    # Tweak to allow module to import local modules
    try:
        sys.path.index(src)
    except:
        sys.path.append(src)

    handler = cfg.get('handler')
    # Inspect the handler string (<module>.<function name>) and translate it
    # into a function we can execute.
    fn = get_callable_handler_function(src, handler)

    # TODO: look into mocking the ``context`` variable, currently being passed
    # as None.

    start = time.time()
    results = fn(event, None)
    end = time.time()

    print('{0}'.format(results))
    if verbose:
        print('\nexecution time: {:.8f}s\nfunction execution '
              'timeout: {:2}s'.format(end - start, cfg.get('timeout', 15)))


def init(src, minimal=False):
    """Copies template files to a given directory.

    :param str src:
        The path to output the template lambda project files.
    :param bool minimal:
        Minimal possible template files (excludes event.json).
    """

    templates_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'project_templates',
    )
    for filename in os.listdir(templates_path):
        if (minimal and filename == 'event.json') or filename.endswith('.pyc'):
            continue
        dest_path = os.path.join(templates_path, filename)

        if not os.path.isdir(dest_path):
            copy(dest_path, src)


def build(src, requirements=False, local_package=None, config_file_path=None):
    """Builds the file bundle.

    :param config_file_path: path to custom config.yam.file
    :param requirements:
    :param str src:
       The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param str local_package:
        The path to a local package with should be included in the deploy as
        well (and/or is not available on PyPi)
    """
    # Load and parse the config file.
    cfg = read_config_file(config_file_path, src)

    # Get the absolute path to the output directory and create it if it doesn't
    # already exist.
    dist_directory = cfg.get('dist_directory', 'dist')
    path_to_dist = os.path.join(src, dist_directory)
    mkdir(path_to_dist)

    # Combine the name of the Lambda function with the current timestamp to use
    # for the output filename.
    function_name = cfg.get('function_name')
    output_filename = '{}.zip'.format(function_name)
    build_config = defaultdict(**cfg.get('build', {}))
    path_to_temp = mkdtemp(prefix='aws-lambda')
    pip_install_to_target(
        path_to_temp,
        requirements=requirements,
        local_package=local_package,
        **cfg['build']
    )

    # Hack for Zope.
    if 'zope' in os.listdir(path_to_temp):
        print(
            'Zope packages detected; fixing Zope package paths to '
            'make them importable.',
        )
        # Touch.
        with open(os.path.join(path_to_temp, 'zope/__init__.py'), 'wb'):
            pass

    # Gracefully handle whether ".zip" was included in the filename or not.
    output_filename = (
        '{0}.zip'.format(output_filename)
        if not output_filename.endswith('.zip')
        else output_filename
    )

    # Allow definition of source code directories we want to build into our
    # zipped package.
    build_source_directories = build_config.get('source_directories', '')
    build_source_directories = (
        build_source_directories
        if build_source_directories is not None
        else ''
    )
    source_directories = [
        d.strip() for d in build_source_directories.split(',')
    ]

    def filter_ignored_files(file_name):
        ignore_file_path = os.path.join(src, ".lambdaignore")
        if os.path.exists(ignore_file_path):
            with open(ignore_file_path) as ignored:
                ignored_patterns = map(str.strip, ignored.readlines())
            return all(file_name not in glob.glob(entry) for entry in ignored_patterns)
        else:
            return True

    files = []
    listdir = os.listdir(src)
    filtered_files = filter(filter_ignored_files, listdir)
    for filename in filtered_files:
        if os.path.isfile(filename):
            if filename == '.DS_Store':
                continue
            if 'yaml' in filename:
                continue
            if filename == '.lambdaignore':
                continue
            print('Bundling: %r' % filename)
            files.append(os.path.join(src, filename))
        elif os.path.isdir(filename) and filename in source_directories:
            print('Bundling directory: %r' % filename)
            files.append(os.path.join(src, filename))

    # "cd" into `temp_path` directory.
    os.chdir(path_to_temp)
    for f in files:
        if os.path.isfile(f):
            _, filename = os.path.split(f)

            # Copy handler file into root of the packages folder.
            copyfile(f, os.path.join(path_to_temp, filename))
        elif os.path.isdir(f):
            destination_folder = os.path.join(path_to_temp, f[len(src) + 1:])
            copytree(f, destination_folder)

    # Zip them together into a single file.
    # TODO: Delete temp directory created once the archive has been compiled.
    path_to_zip_file = archive('./', path_to_dist, output_filename)
    shutil.rmtree(path_to_temp)
    os.chdir(src)
    return path_to_zip_file


def read_config_file(config_file_path, src):
    if config_file_path:
        path_to_config_file = os.path.join(src, config_file_path)
    else:
        path_to_config_file = os.path.join(src, 'config.yaml')
    cfg = read(path_to_config_file, loader=yaml.load)
    return cfg


def get_callable_handler_function(src, handler):
    """Tranlate a string of the form "module.function" into a callable
    function.

    :param str src:
      The path to your Lambda project containing a valid handler file.
    :param str handler:
      A dot delimited string representing the `<module>.<function name>`.
    """

    # "cd" into `src` directory.
    os.chdir(src)

    module_name, function_name = handler.split('.')
    filename = get_handler_filename(handler)

    path_to_module_file = os.path.join(src, filename)
    module = load_source(module_name, path_to_module_file)
    return getattr(module, function_name)


def get_handler_filename(handler):
    """Shortcut to get the filename from the handler string.

    :param str handler:
      A dot delimited string representing the `<module>.<function name>`.
    """
    module_name, _ = handler.split('.')
    return '{0}.py'.format(module_name)


def _install_packages(path, packages):
    """Install all packages listed to the target directory.

    Ignores any package that includes Python itself and python-lambda as well
    since its only needed for deploying and not running the code

    :param str path:
        Path to copy installed pip packages to.
    :param list packages:
        A list of packages to be installed via pip.
    """

    def _filter_blacklist(package):
        blacklist = ['-i', '#', 'Python==', 'python-lambda==', 'hbi-python-lambda==', 'boto3==', 'tox==', 'pip==',
                     'setuptools', 'virtualenv==', 'click==', 'argparse==', 'botocore==']
        return all(package.startswith(entry) is False for entry in blacklist)

    filtered_packages = filter(_filter_blacklist, packages)
    for package in filtered_packages:
        if package.startswith('-e '):
            package = package.replace('-e ', '')

        print('Installing {package}'.format(package=package))
        pip.main(['install', package, '-t', path, '--ignore-installed'])


def pip_install_to_target(path, requirements=False, local_package=None, **kwargs):
    """For a given active virtualenv, gather all installed pip packages then
    copy (re-install) them to the path provided.

    :param str path:
        Path to copy installed pip packages to.
    :param bool requirements:
        If set, only the packages in the requirements.txt file are installed.
        The requirements.txt file needs to be in the same directory as the
        project which shall be deployed.
        Defaults to false and installs all pacakges found via pip freeze if
        not set.
    :param str local_package:
        The path to a local package with should be included in the deploy as
        well (and/or is not available on PyPi)
    """
    packages = []
    if not requirements:
        print('Gathering pip packages')
        packages.extend(pip.operations.freeze.freeze())
    else:
        if os.path.exists('requirements.txt'):
            print('Gathering requirement packages')
            data = read('requirements.txt')
            packages.extend(data.splitlines())
    if 'remote_packages' in kwargs.keys():
        packages.extend(kwargs['remote_packages'].split(','))

    if not packages:
        print('No dependency packages installed!')

    if local_package is not None:
        if not isinstance(local_package, (list, tuple)):
            local_package = [local_package]
        for l_package in local_package:
            packages.append(l_package)
    _install_packages(path, packages)


def get_role_name(region, account_id, role):
    """Shortcut to insert the `account_id` and `role` into the iam string."""
    prefix = ARN_PREFIXES.get(region, 'aws')
    return 'arn:{0}:iam::{1}:role/{2}'.format(prefix, account_id, role)


client_cache = {}


def get_client(client, cfg, aws_profile=None):
    """Shortcut for getting an initialized instance of the boto3 client."""
    if client not in client_cache:
        if aws_profile:
            log.info('Using aws profile name: {}'.format(aws_profile))
            session = boto3.Session(profile_name=aws_profile)
            client_cache[client] = session.client(client, region_name=cfg.get('region'))
        else:
            client_cache[client] = boto3.client(
                client,
                aws_access_key_id=cfg.get('aws_access_key_id'),
                aws_secret_access_key=cfg.get('aws_secret_access_key'),
                region_name=cfg.get('region'),
            )
    return client_cache[client]


def create_function(cfg, path_to_zip_file, upload_to_s3=False, filename=None, aws_profile=None):
    """Register and upload a function to AWS Lambda."""

    print('Creating your new Lambda function')
    byte_stream = read(path_to_zip_file, binary_file=True)
    role = create_role_for_function(cfg, aws_profile=aws_profile)

    client = get_client('lambda', cfg, aws_profile=aws_profile)

    # Do we prefer development variable over config?
    func_name = (
        os.environ.get('LAMBDA_FUNCTION_NAME') or cfg.get('function_name')
    )
    print('Creating lambda function with name: {}'.format(func_name))
    code_params_dict = {}
    code_params_dict.update([('ZipFile', byte_stream)]) if not upload_to_s3 else code_params_dict.update(
        [('S3Bucket', cfg.get('bucket_name')), ('S3Key', filename)]
    )
    kwargs = {
        'FunctionName': func_name,
        'Runtime': cfg.get('runtime', 'python2.7'),
        'Role': role,
        'Handler': cfg.get('handler'),
        'Code': code_params_dict,
        'Description': cfg.get('description'),
        'Timeout': cfg.get('timeout', 15),
        'MemorySize': cfg.get('memory_size', 512),
        'Publish': True,
    }

    if 'environment_variables' in cfg and cfg.get('environment_variables'):
        kwargs.update(
            Environment={
                'Variables': {
                    key: get_environment_variable_value(value)
                    for key, value
                    in cfg.get('environment_variables').items()
                },
            },
        )
    for i in range(5):
        try:
            if function_exists(cfg, func_name, aws_profile=aws_profile):
                continue
            else:
                client.create_function(**kwargs)
                print('Successfully created function {}'.format(func_name))
        except Exception as e:
            print("Error while updating function, backing off.")
            time.sleep(5)  # aws tells that deploys everything almost immediately. Almost...
            if i > 3:
                raise e


def update_function(cfg, path_to_zip_file, upload_to_s3=False, filename=None, aws_profile=None):
    """Updates the code of an existing Lambda function"""

    print('Updating your Lambda function')
    byte_stream = read(path_to_zip_file, binary_file=True)

    role = create_role_for_function(cfg, aws_profile=aws_profile)
    client = get_client('lambda', cfg, aws_profile=aws_profile)
    if not upload_to_s3:
        client.update_function_code(
            FunctionName=cfg.get('function_name'),
            ZipFile=byte_stream,
            Publish=False,
        )
    else:
        client.update_function_code(
            FunctionName=cfg.get('function_name'),
            S3Bucket=cfg.get('bucket_name'),
            S3Key=filename,
            Publish=False,
        )
    kwargs = {
        'FunctionName': cfg.get('function_name'),
        'Role': role,
        'Handler': cfg.get('handler'),
        'Description': cfg.get('description'),
        'Timeout': cfg.get('timeout', 15),
        'MemorySize': cfg.get('memory_size', 512),
        'VpcConfig': {
            'SubnetIds': cfg.get('subnet_ids', []),
            'SecurityGroupIds': cfg.get('security_group_ids', []),
        },
    }

    if 'environment_variables' in cfg:
        kwargs.update(
            Environment={
                'Variables': {
                    key: get_environment_variable_value(value)
                    for key, value
                    in cfg.get('environment_variables').items()
                },
            },
        )
    for i in range(5):
        try:
            if client.update_function_configuration(**kwargs):
                print("Successfully updated function {}".format(cfg.get('function_name')))
                break
        except Exception as e:
            print("Error while updating function, backing off.")
            time.sleep(5)  # aws tells that deploys everything almost immediately. Almost...
            if i > 3:
                raise e

    # Publish last, so versions pick up eventually updated description...
    client.publish_version(
        FunctionName=cfg.get('function_name'),
    )


def create_role_for_function(cfg, aws_profile=None):
    role_cfg = cfg.get('role')
    if role_cfg is not None:
        if not get_role_arn(role_cfg['name'], cfg, aws_profile=aws_profile):
            log.info("Creating new role: {}".format(role_cfg['name']))
            role = create_role(role_cfg['name'], cfg, aws_profile=aws_profile)
        else:
            log.info("Found an existing role, updating policies")
            role = get_role_arn(role_cfg['name'], cfg, aws_profile=aws_profile)
            put_role_policy(role_cfg['name'], cfg, aws_profile=aws_profile)
    else:
        log.info("""No roles found. Will use role with name: lambda_basic_execution.\n
                 You can create one by updating your configuration and calling $lambda deploy.""")
        role = get_role_arn("lambda_basic_execution", cfg=cfg, aws_profile=aws_profile)
    return role


def upload_s3(cfg, path_to_zip_file, aws_profile=None):
    """Upload a function to AWS S3."""

    print('Uploading your new Lambda function')
    client = get_client('s3', cfg, aws_profile=aws_profile)
    byte_stream = b''
    with open(path_to_zip_file, mode='rb') as fh:
        byte_stream = fh.read()
    s3_key_prefix = cfg.get('s3_key_prefix', '/dist')
    filename = '{prefix}.zip'.format(prefix=s3_key_prefix)

    # Do we prefer development variable over config?
    buck_name = (
        os.environ.get('S3_BUCKET_NAME') or cfg.get('bucket_name')
    )
    func_name = (
        os.environ.get('LAMBDA_FUNCTION_NAME') or cfg.get('function_name')
    )
    kwargs = {
        'Bucket': '{}'.format(buck_name),
        'Key': '{}'.format(filename),
        'Body': byte_stream,
    }

    client.put_object(**kwargs)
    print('Finished uploading {} to S3 bucket {}'.format(func_name, buck_name))
    return filename


def function_exists(cfg, function_name, aws_profile=None):
    """Check whether a function exists or not"""
    client = get_client('lambda', cfg, aws_profile=aws_profile)

    # Need to loop through until we get all of the lambda functions returned.
    # It appears to be only returning 50 functions at a time.
    functions = []
    functions_resp = client.list_functions()
    functions.extend([
        f['FunctionName'] for f in functions_resp.get('Functions', [])
    ])
    while 'NextMarker' in functions_resp:
        functions_resp = client.list_functions(
            Marker=functions_resp.get('NextMarker'),
        )
        functions.extend([
            f['FunctionName'] for f in functions_resp.get('Functions', [])
        ])
    return function_name in functions


def create_trigger(cfg, aws_profile=None):
    """Creates trigger and associates it with function function (S3 or CloudWatch)"""
    trigger_type = cfg.get('trigger')['type']
    log.info("Creating trigger: {}".format(trigger_type))
    return {
        "bucket": create_trigger_s3,
        "event": create_trigger_cloud_watch,
        "sns": create_sns_trigger
    }[trigger_type](cfg, aws_profile)


def create_trigger_s3(cfg, aws_profile=None):
    s3_client = get_client('s3', cfg, aws_profile=aws_profile)
    bucket_notification = s3_client.BucketNotification(cfg.get('trigger')['bucket_name'])
    bucket_notification.put(
        NotificationConfiguration={
            'LambdaFunctionConfigurations': [
                {
                    'LambdaFunctionArn': get_function_arn_name(cfg, aws_profile=aws_profile),
                    'Events': cfg.get('trigger')['events']
                }
            ]
        }
    )


def create_trigger_cloud_watch(cfg, aws_profile=None):
    """Creates or updates cron trigger and associates it with lambda function"""
    lambda_client = get_client('lambda', cfg, aws_profile=aws_profile)
    events_client = get_client('events', cfg, aws_profile=aws_profile)
    function_arn = get_function_arn_name(cfg, aws_profile=aws_profile)
    frequency = cfg.get('trigger')['frequency']
    trigger_name = "{}".format(cfg.get('trigger')['name'])

    rule_response = events_client.put_rule(
        Name=trigger_name,
        ScheduleExpression=frequency,
        State='DISABLED'
    )

    statement_id = "{}-Event".format(trigger_name)
    try:
        lambda_client.remove_permission(
            FunctionName=function_arn,
            StatementId=statement_id,
        )
    except Exception:  # sanity check if resource is not found. boto uses its own factory to instantiate exceptions
        pass  # that's why exception clause is so broad

    lambda_client.add_permission(
        FunctionName=function_arn,
        StatementId=statement_id,
        Action="lambda:InvokeFunction",
        Principal="events.amazonaws.com",
        SourceArn=rule_response['RuleArn']
    )

    events_client.put_targets(
        Rule=trigger_name,
        Targets=[
            {
                'Id': "1",
                'Arn': function_arn
            }
        ]
    )


def create_sns_trigger(cfg, aws_profile=None):
    sns_client = get_client('sns', cfg, aws_profile)
    lambda_client = get_client('lambda', cfg, aws_profile)
    s3_client = get_client('s3', cfg, aws_profile)

    function_arn = get_function_arn_name(cfg, aws_profile)
    trigger_name = cfg.get('trigger')['name']
    topic_arn = sns_client.create_topic(
        Name=trigger_name
    )['TopicArn']

    topic_policy_document = """
        {{
  "Version": "2008-10-17",
  "Id": "__default_policy_ID",
  "Statement": [
    {{
      "Sid": "_s3",
      "Effect": "Allow",
      "Principal": {{
        "Service": "s3.amazonaws.com"
      }},
      "Action": [
        "SNS:Publish"
      ],
      "Resource": "{topic_arn}",
      "Condition": {{
        "StringLike": {{
          "aws:SourceArn": "arn:aws:s3:::*"
        }}
      }}
    }}
  ]
}}"""
    sns_client.set_topic_attributes(
        TopicArn=topic_arn,
        AttributeName='Policy',
        AttributeValue=topic_policy_document.format(topic_arn=topic_arn)
    )

    sns_client.subscribe(
        TopicArn=topic_arn,
        Protocol='lambda',
        Endpoint=function_arn
    )

    statement_id = "{}-Topic".format(trigger_name)
    try:
        lambda_client.remove_permission(
            FunctionName=function_arn,
            StatementId=statement_id,
        )
    except Exception:  # sanity check if resource is not found. boto uses its own factory to instantiate exceptions
        pass  # that's why exception clause is so broad
    lambda_client.add_permission(
        FunctionName=function_arn,
        StatementId=statement_id,
        Action="lambda:InvokeFunction",
        Principal="sns.amazonaws.com",
        SourceArn=topic_arn
    )
    for bucket in cfg.get('trigger')['buckets']:
        bucket_values = bucket.values()[0]
        s3_client.put_bucket_notification_configuration(
            Bucket=bucket_values['bucket_name'],
            NotificationConfiguration={
                'TopicConfigurations': [
                    {
                        'TopicArn': topic_arn,
                        'Events': bucket_values['events'],
                        'Filter': {
                            'Key': {
                                'FilterRules': [
                                    {
                                        'Name': 'prefix',
                                        'Value': bucket_values['prefix']
                                    },
                                    {
                                        'Name': 'suffix',
                                        'Value': bucket_values['suffix']
                                    }
                                ]
                            }
                        }
                    }
                ]
            }
        )


def get_function_arn_name(cfg, aws_profile):
    """Retrieves arn name of an existing function"""
    client = get_client('lambda', cfg, aws_profile=aws_profile)
    return client.get_function(FunctionName=cfg.get('function_name'))['Configuration']['FunctionArn']


def get_role_arn(role_name, cfg, aws_profile=None):
    client = get_client("iam", cfg, aws_profile=aws_profile)
    response = None
    try:
        response = client.get_role(
            RoleName=role_name
        )['Role']['Arn']
    except Exception as e:
        pass
    return response


def create_role(role_name, cfg, aws_profile=None):
    client = get_client('iam', cfg, aws_profile=aws_profile)
    response = client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument="""{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    },
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "apigateway.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}"""
    )
    role_arn = response['Role']['Arn']
    put_role_policy(role_name, cfg, aws_profile)
    print("Checking if policy is available.")
    policy = client.get_role_policy(RoleName=role_name, PolicyName=cfg.get('role')['policy_name'])
    assert policy['ResponseMetadata']['HTTPStatusCode'] == 200
    return role_arn


def put_role_policy(role_name, cfg, aws_profile=None):
    client = get_client('iam', cfg, aws_profile=aws_profile)
    role_cfg = cfg.get('role')
    if os.path.exists(os.path.join(os.getcwd(), role_cfg['policy_document'])):
        try:
            with open(role_cfg['policy_document']) as policy:
                client.put_role_policy(
                    RoleName=role_name,
                    PolicyName=role_cfg['policy_name'],
                    PolicyDocument=json.dumps(json.load(policy))
                )
        except Exception as e:
            log.warn(e.message)
    else:
        log.debug("No policy file found")
