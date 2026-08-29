from argparse import ArgumentParser

from nbcore.resource_utils.utils import load_notebook, read_json

from .converter import convert_to_script


def compile_notebook(filename: str) -> str:
    notebook = read_json(filename)
    cells = load_notebook(notebook["cells"])
    return convert_to_script(cells)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="NBCompile - notebook to script converter")
    parser.add_argument("-f", "--filename", type=str, required=True, help="Path to the notebook (.ipynb) to convert.")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Write the script here instead of printing it to stdout.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    script = compile_notebook(args.filename)

    if args.output:
        with open(args.output, "w") as f:
            f.write(script)
    else:
        print(script)


if __name__ == "__main__":
    main()
