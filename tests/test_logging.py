from qsarcons.logging import FailedMolecule, OutputSuppressor


def test_failed_molecule_repr_includes_smiles_and_message():
    fm = FailedMolecule("bad_smiles!!!", message="SMILES parsing failed")
    assert repr(fm) == "bad_smiles!!! -> SMILES parsing failed"


def test_failed_molecule_default_message():
    fm = FailedMolecule("x")
    assert fm.message == "failed"


def test_output_suppressor_silences_stdout(capfd):
    with OutputSuppressor():
        print("should not appear")
    assert capfd.readouterr().out == ""


def test_output_suppressor_nests_safely(capfd):
    """Nested use restores output only once the outermost context exits."""
    with OutputSuppressor():
        with OutputSuppressor():
            print("nested")
        print("still suppressed")
    assert capfd.readouterr().out == ""


def test_output_suppressor_restores_stdout_after_exit(capfd):
    with OutputSuppressor():
        print("suppressed")
    print("visible")
    assert capfd.readouterr().out == "visible\n"
