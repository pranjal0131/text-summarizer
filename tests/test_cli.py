import pytest

from summarizer.cli import main

SAMPLE = (
    "Graphs model relationships between entities. "
    "PageRank ranks graph nodes by importance. "
    "Cooking pasta requires boiling water."
)


def test_summarize_text_argument(capsys):
    exit_code = main(["--text", SAMPLE, "--sentences", "1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip()


def test_summarize_from_file(tmp_path, capsys):
    source = tmp_path / "input.txt"
    source.write_text(SAMPLE, encoding="utf-8")
    exit_code = main(["--file", str(source), "--sentences", "2"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip()


def test_show_scores_lists_every_sentence(capsys):
    main(["--text", SAMPLE, "--show-scores"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert all(line.startswith("[") for line in lines)


def test_frequency_method(capsys):
    exit_code = main(["--text", SAMPLE, "--method", "frequency", "--sentences", "1"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip()


def test_empty_text_fails(capsys):
    exit_code = main(["--text", "   "])
    assert exit_code == 1
    assert "empty" in capsys.readouterr().err


def test_unknown_method_rejected():
    with pytest.raises(SystemExit):
        main(["--text", SAMPLE, "--method", "nonexistent"])
