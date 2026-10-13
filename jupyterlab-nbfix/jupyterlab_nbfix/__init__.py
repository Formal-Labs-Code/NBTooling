def _jupyter_labextension_paths():
    # Must match pyproject.toml's [tool.hatch.build.targets.wheel.shared-data]
    # destination ("jupyterlab-nbfix", the actual extension/package name) -
    # this is the separate declaration `jupyter labextension develop`
    # (symlink-based dev installs) reads; a plain `pip install` reads the
    # pyproject.toml one instead. Both must agree or a dev install ends up
    # symlinked under the wrong name (found via a real dev-install attempt
    # creating share/jupyter/labextensions/nbfix instead of .../jupyterlab-nbfix).
    return [{"src": "labextension", "dest": "jupyterlab-nbfix"}]
