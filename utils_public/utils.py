import subprocess


def run_cmd(
    cmd_and_args: list[str], shell: bool = False, raise_on_fail: bool = True
) -> subprocess.CompletedProcess:
    """Runs a shell command.
    Prints shell command's stdout/stderr as well to stdout.
    This is useful for Jupyter notebooks, which seems to suppress the stdout/stderr when
    directly calling shell commands via `!cmd arg0`.

    Args:
        cmd_and_args (list[str]): Shell command, following structure:
            [str command, str arg0, str arg1, ...]
            Examples:
                ["ls"]
                ["cp", "src", "dst"]
        shell (bool): Run command directly via shell.
        raise_on_fail (bool): if True, this will raise a RuntimeError() if the command fails.

    Returns:
        subprocess.CompletedProcess: result of calling `subprocess.run()`.
    """
    result = subprocess.run(
        cmd_and_args, capture_output=True, text=True, check=False, shell=shell
    )
    print(f"(stdout):\n{result.stdout}")
    print(f"(stderr):\n{result.stderr}")
    if result.returncode != 0:
        print(f"Error: command {result.args} failed")
        if raise_on_fail:
            raise RuntimeError(f"Error: command {result.args} failed")
    return result
