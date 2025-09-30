import pytest
import textwrap
from llmjj.prompt_builder import PromptBuilder

def _write_config(tmp_path, contents: str):
    path = tmp_path / "prompt.yaml"
    path.write_text(contents)
    return path


def test_prompt_builder_renders_sample_prompt(tmp_path):
    config = textwrap.dedent(
        """
        template: |-
          {persona_instruction}

          ## Query (id={query_id})
          {query}

          ## Document
          {document}
        fields:
          persona_instruction: You are an AI judge replicating MS MARCO annotators.
          query_id: query-123
          query: What is YAML?
        """
    ).strip()
    config_path = _write_config(tmp_path, config)

    builder = PromptBuilder(str(config_path))
    prompt = builder.build({"document": "YAML is a human-friendly serialization format."})

    assert "You are an AI judge replicating MS MARCO annotators." in prompt
    assert "## Query (id=query-123)" in prompt
    assert "YAML is a human-friendly serialization format." in prompt


def test_build_merges_static_and_dynamic_fields(tmp_path):
    config = textwrap.dedent(
        """
        template: "Hello {name}! Source: {source}."
        fields:
          source: config-file
        """
    ).strip()
    config_path = _write_config(tmp_path, config)

    builder = PromptBuilder(str(config_path))

    prompt = builder.build({"name": "Alice"})

    assert prompt == "Hello Alice! Source: config-file."


def test_dynamic_fields_override_static_defaults(tmp_path):
    config = textwrap.dedent(
        """
        template: "{greeting}, {name}!"
        fields:
          greeting: Hello
          name: Default
        """
    ).strip()
    config_path = _write_config(tmp_path, config)

    builder = PromptBuilder(str(config_path))

    prompt = builder.build({"name": "Bob", "greeting": "Howdy"})

    assert prompt == "Howdy, Bob!"
