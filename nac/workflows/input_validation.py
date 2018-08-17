from jsonschema import (Draft4Validator, validate,  validators)
from typing import Dict
import json
import jsonref
import yaml
import pkg_resources as pkg

schema_workflows = {
    'absorption_spectrum': pkg.resource_filename("nac", "data/schemas/absorption_spectrum.json"),
    'derivative_couplings': pkg.resource_filename("nac", "data/schemas/derivative_couplings.json"),
    'general_settings': pkg.resource_filename("nac", "data/schemas/general_settings.json")}


def process_input(input_file: str, workflow_name) -> Dict:
    """
    Read the input file in YAML format, validate it again the schema
    of `workflow_name` and return a nested dictionary with the input.
    """
    input_dict = read_json_yaml(input_file, fmt='yaml')
    path_schema = schema_workflows[workflow_name]
    schema = load_json_schema(path_schema)

    return validate_input(input_dict, schema)


def extend_with_default(validator_class):
    """ Extend the json schema validator so it fills in the defaults"""
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(prop, subschema["default"])

        for error in validate_properties(
                validator, properties, instance, schema):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults})


def validate_input(input_dict: dict, schema: Dict) -> Dict:
    """
    Check that the input is correct following `schema` and get
    the default values from the schema.
    """
    # first validate input
    validate(input_dict, schema)

    # Next add defaults
    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    DefaultValidatingDraft4Validator(schema).validate(input_dict)

    return input_dict


def load_json_schema(file_path: str) -> Dict:
    "Load a schema from `file_path` and use the absolute path for file references"

    # Absolute path prefix
    root = pkg.resource_filename('nac', 'data')

    base_uri = "file://{}/".format(root)

    with open(file_path, 'r') as f:
        xs = f.read()

    # replace ref with absolute values to the files
    return jsonref.loads(xs, base_uri=base_uri, jsonschema=True)


def read_json_yaml(input_file: str, fmt: str) -> Dict:
    """
    Read a file in json or yaml format.
    """
    mod = yaml if fmt is 'yaml' else json
    with open(input_file, 'r') as f:
        xs = mod.load(f)

    return xs
