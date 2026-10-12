import json

from nbcore.resource_utils.utils import TEST_RES_PATH, read_json

from nbharness.mcp_server import analyze


notebook = read_json(TEST_RES_PATH + "dataleak_true.ipynb")

print(json.dumps({
    "analyze": analyze(notebook),
}, indent=2))
