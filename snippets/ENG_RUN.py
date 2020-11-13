source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd()
spec = spec_from_file_location("three_d_model", cwd / 'three_d_model.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
foo.fit_3_d_model_slice_input(slice(0, 10))
"""
if __name__ == '__main__':
    exec(source_code)
