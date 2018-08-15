from nac.workflows.input_validation import (read_json_yaml, schema_workflows)
import jsonschema

path = "test/test_files/input_test_oscillator.yml"

inp = read_json_yaml(path, fmt='yaml')
schema = schema_workflows['absorptionspectrum']

jsonschema.validate(inp, schema)
