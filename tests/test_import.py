def test_import():
    import k_openvino

    assert k_openvino is not None


def test_cli_help():
    from typer.testing import CliRunner

    from k_openvino.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "openvino" in result.output.lower()
