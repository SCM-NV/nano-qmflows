from jsonschema import Draft4Validator, validators
from typing import Dict
import json
import yaml
import pkg_resources as pkg

schema_workflows = {
    'absorptionspectrum': json.loads(pkg.resource_string("nac", "data/schemas/absorption_spectrum.json").decode()),
    'general_settings': json.loads(pkg.resource_string("nac", "data/schemas/general_settings.json").decode())}


def process_input(input_file: str, workflow_name) -> Dict:
    """
    Read the input file in YAML format, validate it again the schema
    of `workflow_name` and return a nested dictionary with the input.
    """
    input_dict = read_json_yaml(input_file, fmt='json')
    path_schema = schema_workflows['workflow_name']
    schema = read_json_yaml(path_schema, fmt='yaml')

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
    DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)
    return DefaultValidatingDraft4Validator(schema).validate(input_dict)


def read_json_yaml(input_file: str, fmt: str) -> Dict:
    """
    Read a file in json or yaml format.
    """
    mod = yaml if fmt is 'yaml' else json
    with open(input_file, 'r') as f:
        xs = mod.load(f)

    return xs
