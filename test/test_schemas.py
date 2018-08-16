from nac.workflows.input_validation import (load_json_schema, read_json_yaml, schema_workflows)
import jsonschema


def test_input_validation():
    """
    Test the input validation schema
    """
    path = "test/test_files/input_test_oscillator.yml"
    inp = read_json_yaml(path, fmt='yaml')
    schema = load_json_schema(schema_workflows['general_settings'])
    jsonschema.validate(inp['general_settings'], schema)
