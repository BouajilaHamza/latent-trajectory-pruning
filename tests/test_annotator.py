"""Tests for the annotator module's pure functions (no API/model calls)."""

from src.annotator import extract_boxed, is_correct, load_env_vars


class TestExtractBoxed:
    """Tests for balanced-brace \\boxed{} extraction."""

    def test_simple_boxed(self):
        assert extract_boxed("The answer is \\boxed{42}.") == "42"

    def test_nested_braces(self):
        assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_multiple_boxed_returns_last(self):
        text = "First \\boxed{wrong}, then \\boxed{correct}"
        assert extract_boxed(text) == "correct"

    def test_no_boxed_returns_stripped(self):
        assert extract_boxed("  just text  ") == "just text"

    def test_deeply_nested(self):
        assert extract_boxed("\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}") == (
            "\\left(3, \\frac{\\pi}{2}\\right)"
        )

    def test_unclosed_boxed(self):
        # If the closing brace is missing, return everything after \boxed{
        result = extract_boxed("\\boxed{42")
        assert result == "42"


class TestIsCorrect:
    """Tests for trajectory correctness checking."""

    def test_matching_boxed_answers(self):
        trajectory = "Step 1... Step 2... \\boxed{42}"
        answer = "The solution is \\boxed{42}"
        assert is_correct(trajectory, answer)

    def test_mismatched_boxed_answers(self):
        trajectory = "Step 1... \\boxed{41}"
        answer = "\\boxed{42}"
        assert not is_correct(trajectory, answer)

    def test_no_boxed_in_trajectory(self):
        # When no \boxed in trajectory, extract_boxed returns the full stripped
        # text which won't match the answer's boxed content
        trajectory = "The answer is 42"
        answer = "\\boxed{42}"
        # extract_boxed(trajectory) = "The answer is 42" != "42"
        assert not is_correct(trajectory, answer)

    def test_both_no_boxed(self):
        # Both return full stripped text; if they match it's "correct"
        trajectory = "42"
        answer = "42"
        assert is_correct(trajectory, answer)


class TestLoadEnvVars:
    """Tests for .env file loading."""

    def test_loads_from_nonexistent_file(self):
        # Should not crash, returns os.environ merged
        result = load_env_vars("/nonexistent/.env")
        assert isinstance(result, dict)

    def test_loads_from_actual_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=test_value\n# comment\nOTHER=123\n")
        result = load_env_vars(str(env_file))
        assert result["TEST_KEY"] == "test_value"
        assert result["OTHER"] == "123"

    def test_skips_comments_and_empty_lines(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=val\n")
        result = load_env_vars(str(env_file))
        assert result["KEY"] == "val"
        assert "# comment" not in result
