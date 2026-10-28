import unittest
from unittest.mock import patch, MagicMock
from urllib.error import URLError, HTTPError
import tempfile
from pathlib import Path

from app import (
    ask_ollama,
    build_research_prompt,
    cosine_similarity,
    create_contextual_sourced_chunks,
    create_sourced_chunks,
    create_web_chunks,
    extract_keywords,
    get_source_names,
    load_text_documents,
    retrieve_relevant_chunks,
    retrieve_semantic_chunks,
    get_source_references,
    clean_research_response,
    save_report,
    list_saved_reports,
    read_saved_report,
    load_config,
    run_research
)

# Run this module with `python -m unittest test_app.py`, or discover all tests
# with `python -m unittest discover`.

class TestLocalResearchAssistant(unittest.TestCase):
    """Verify research helpers, service error handling, and report storage."""

    def test_ask_ollama_is_available(self):
        """Confirm the Ollama generation helper is importable and callable."""
        self.assertTrue(callable(ask_ollama))

    # Stub the network call so these tests stay local and deterministic.
    @patch("app.urlopen")
    def test_ask_ollama_returns_response(self, mock_urlopen):
        """Return the response text when Ollama sends a valid payload."""
        fake_response = MagicMock()
        fake_response.read.return_value = (
            b'{"response": "This is a test response."}'
        )

        mock_urlopen.return_value.__enter__.return_value = fake_response

        result = ask_ollama("Test question")

        self.assertEqual(result, "This is a test response.")    

    @patch("app.urlopen")
    def test_ask_ollama_handles_connection_error(self, mock_urlopen):
        """Convert an Ollama connection failure into a readable message."""
        mock_urlopen.side_effect = URLError("Connection refused")

        result = ask_ollama("Test question")

        self.assertEqual(
            result,
            "Could not connect to Ollama: Connection refused"
        )                

    @patch("app.urlopen")
    def test_ask_ollama_handles_missing_response_field(self, mock_urlopen):
        """Report an unexpected response when Ollama omits its text field."""
        fake_response = MagicMock()
        fake_response.read.return_value = b'{"unexpected": "value"}'

        mock_urlopen.return_value.__enter__.return_value = fake_response

        result = ask_ollama("Test question")

        self.assertEqual(
            result,
            "Error: Ollama returned an unexpected response."
        )

    @patch("app.urlopen")
    def test_ask_ollama_handles_invalid_json(self, mock_urlopen):
        """Report an unexpected response when Ollama returns invalid JSON."""
        fake_response = MagicMock()
        fake_response.read.return_value = b"This is not valid JSON"

        mock_urlopen.return_value.__enter__.return_value = fake_response

        result = ask_ollama("Test question")

        self.assertEqual(
            result,
            "Error: Ollama returned an unexpected response."
        )

    @patch("app.urlopen")
    def test_ask_ollama_handles_http_error(self, mock_urlopen):
        """Include the HTTP status code when the Ollama API rejects a call."""
        mock_urlopen.side_effect = HTTPError(
            url="http://localhost:11434/api/generate",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None
        )

        result = ask_ollama("Test question")

        self.assertEqual(
            result,
            "Ollama API error: HTTP 500"
        )

    def test_build_research_prompt_contains_question_and_evidence(self):
        """Include both the user's question and retrieved evidence in the prompt."""
        question = "What are the benefits of local AI?"
        evidence = "Local AI can keep sensitive data on local infrastructure."

        prompt = build_research_prompt(question, evidence)

        self.assertIn(question, prompt)
        self.assertIn(evidence, prompt)

    def test_extract_keywords_removes_stop_words(self):
        """Remove common stop words while keeping useful question terms."""
        keywords = extract_keywords(
            "What are the advantages of local AI?"
        )

        self.assertEqual(
            keywords,
            {"advantages", "local", "ai"}
        )


    def test_retrieve_relevant_chunks_ranks_best_match_first(self):
        """Place the chunk with the greatest keyword overlap at the top."""
        chunks = [
            {
                "source": "cloud.txt",
                "content": "Cloud services can scale without local hardware."
            },
            {
                "source": "local.txt",
                "content": "Local AI provides local control and privacy advantages."
            },
            {
                "source": "hardware.txt",
                "content": "Large models may require significant computing resources."
            }
        ]

        results = retrieve_relevant_chunks(
            "What are the advantages of local AI?",
            chunks
        )

        self.assertEqual(
            results[0]["source"],
            "local.txt"
        )

        self.assertEqual(
            results[0]["content"],
            "Local AI provides local control and privacy advantages."
        )

    def test_extract_keywords_normalizes_aliases(self):
        """Normalize configured synonyms before returning extracted keywords."""
        keywords = extract_keywords(
            "What are the benefits of running AI privately?"
        )

        self.assertIn("advantages", keywords)
        self.assertIn("privacy", keywords)
        self.assertNotIn("benefits", keywords)
        self.assertNotIn("privately", keywords)

    def test_retrieve_relevant_chunks_respects_top_k(self):
        """Limit keyword retrieval results to the requested number of chunks."""
        chunks = [
            {
                "source": "one.txt",
                "content": "Local AI offers privacy advantages."
            },
            {
                "source": "two.txt",
                "content": "Local AI provides greater local control."
            },
            {
                "source": "three.txt",
                "content": "Local AI can run without a cloud provider."
            },
            {
                "source": "four.txt",
                "content": "Local AI may require additional hardware."
            }
        ]

        results = retrieve_relevant_chunks(
            "What are the advantages of local AI?",
            chunks,
            top_k=2
        )

        self.assertEqual(len(results), 2)



    def test_load_text_documents_loads_only_txt_files(self):
        """Load text files in filename order while ignoring other extensions."""
        with tempfile.TemporaryDirectory() as temporary_folder:
            folder = Path(temporary_folder)

            (folder / "first.txt").write_text(
                "First research document.",
                encoding="utf-8"
            )

            (folder / "second.txt").write_text(
                "Second research document.",
                encoding="utf-8"
            )

            (folder / "ignored.md").write_text(
                "This file should not be loaded.",
                encoding="utf-8"
            )

            documents = load_text_documents(folder)

            self.assertEqual(len(documents), 2)
            self.assertEqual(documents[0]["source"], "first.txt")
            self.assertEqual(
                documents[0]["content"],
                "First research document."
            )
            self.assertEqual(documents[1]["source"], "second.txt")

    def test_create_sourced_chunks_retains_source(self):
        """Keep each original filename attached to its paragraph chunks."""
        documents = [
            {
                "source": "local_ai.txt",
                "content": (
                    "Local AI offers privacy advantages.\n\n"
                    "Local AI may require additional hardware."
                )
            },
            {
                "source": "cloud_ai.txt",
                "content": "Cloud AI can provide easier scalability."
            }
        ]

        chunks = create_sourced_chunks(documents)

        self.assertEqual(len(chunks), 3)

        self.assertEqual(chunks[0]["source"], "local_ai.txt")
        self.assertEqual(
            chunks[0]["content"],
            "Local AI offers privacy advantages."
        )

        self.assertEqual(chunks[1]["source"], "local_ai.txt")
        self.assertEqual(chunks[2]["source"], "cloud_ai.txt")


    def test_retrieval_selects_correct_source_across_documents(self):
        """Return the source whose paragraph best matches the research query."""
        documents = [
            {
                "source": "local_ai.txt",
                "content": (
                    "Local AI runs on infrastructure controlled by the user.\n\n"
                    "Local AI can provide privacy advantages."
                )
            },
            {
                "source": "cloud_ai.txt",
                "content": (
                    "Cloud AI runs on infrastructure managed by a provider.\n\n"
                    "Cloud AI can scale computing resources based on demand."
                )
            }
        ]

        chunks = create_sourced_chunks(documents)

        results = retrieve_relevant_chunks(
            "How can cloud AI scale computing resources?",
            chunks,
            top_k=1
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "cloud_ai.txt")
        self.assertIn(
            "scale computing resources",
            results[0]["content"]
        )

    def test_get_source_names_returns_unique_sources(self):
        """Return each cited filename once in sorted order."""
        chunks = [
            {
                "source": "ai_notes.txt",
                "content": "Local AI provides privacy."
            },
            {
                "source": "cloud_ai_notes.txt",
                "content": "Cloud AI provides scalability."
            },
            {
                "source": "ai_notes.txt",
                "content": "Local AI requires hardware."
            }
        ]

        sources = get_source_names(chunks)

        self.assertEqual(
            sources,
            ["ai_notes.txt", "cloud_ai_notes.txt"]
        )

    @patch("app.get_embedding")
    def test_semantic_retrieval_selects_most_similar_chunk(
        self,
        mock_get_embedding
    ):
        """Rank the locally relevant chunk above a semantically weak result."""
        chunks = [
            {
                "source": "cloud.txt",
                "content": "Cloud services scale resources on demand."
            },
            {
                "source": "local.txt",
                "content": (
                    "Running models locally provides control "
                    "over confidential data."
                )
            }
        ]

        def fake_embedding(text, input_type="document"):
            """Provide deterministic vectors for the semantic-ranking test."""
            if input_type == "query":
                return [1.0, 0.0]

            if "Cloud services" in text:
                return [0.0, 1.0]

            return [0.9, 0.1]

        mock_get_embedding.side_effect = fake_embedding

        results = retrieve_semantic_chunks(
            (
                "How can a company protect private information "
                "using its own infrastructure?"
            ),
            chunks,
            top_k=1
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "local.txt")

    @patch("app.get_embedding")
    def test_semantic_retrieval_excludes_low_similarity_chunks(
        self,
        mock_get_embedding
    ):
        """Exclude chunks whose cosine similarity is below the threshold."""
        chunks = [
            {
                "source": "unrelated.txt",
                "content": "This evidence is unrelated."
            }
        ]

        def fake_embedding(text, input_type="document"):
            """Return orthogonal vectors so the evidence fails the threshold."""
            if input_type == "query":
                return [1.0, 0.0]

            return [0.0, 1.0]

        mock_get_embedding.side_effect = fake_embedding

        results = retrieve_semantic_chunks(
            "A research question",
            chunks,
            min_similarity=0.55
        )

        self.assertEqual(results, [])

    def test_contextual_chunks_include_adjacent_paragraphs(self):
        """Include adjacent paragraphs in each overlapping contextual chunk."""
        documents = [
            {
                "source": "local_ai.txt",
                "content": (
                    "Local AI runs on user-controlled hardware.\n\n"
                    "Prompts and documents can remain local.\n\n"
                    "Large models may require more computing resources."
                )
            }
        ]

        chunks = create_contextual_sourced_chunks(
            documents,
            window_size=2
        )

        self.assertEqual(len(chunks), 3)

        self.assertEqual(
            chunks[0]["source"],
            "local_ai.txt"
        )

        self.assertIn(
            "Local AI runs on user-controlled hardware.",
            chunks[0]["content"]
        )

        self.assertIn(
            "Prompts and documents can remain local.",
            chunks[0]["content"]
        )

        self.assertNotIn(
            "Large models may require more computing resources.",
            chunks[0]["content"]
        )

    def test_create_web_chunks_retains_url_and_source(self):
        """Preserve page title and URL when converting web results to chunks."""
        web_results = [
            {
                "title": "Example Research Page",
                "url": "https://example.com/research",
                "content": "Example evidence about local AI."
            }
        ]

        chunks = create_web_chunks(web_results)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0]["source"],
            "Example Research Page"
        )
        self.assertEqual(
            chunks[0]["url"],
            "https://example.com/research"
        )
        self.assertEqual(
            chunks[0]["content"],
            "Example evidence about local AI."
        )
        self.assertEqual(
            chunks[0]["source_type"],
            "web"
        )

    def test_get_source_references_preserves_web_urls(self):
        """Keep web URLs and local citation types in the source references."""
        chunks = [
            {
                "source": "Local notes",
                "content": "Local evidence.",
                "source_type": "local",
                "url": None
            },
            {
                "source": "Example Web Research",
                "content": "Web evidence.",
                "source_type": "web",
                "url": "https://example.com/research"
            }
        ]

        sources = get_source_references(chunks)

        self.assertEqual(len(sources), 2)

        self.assertEqual(
            sources[0]["type"],
            "local"
        )

        self.assertEqual(
            sources[1]["type"],
            "web"
        )

        self.assertEqual(
            sources[1]["url"],
            "https://example.com/research"
        )

    def test_clean_research_response_removes_prompt_leakage(self):
        """Remove model-generated rationale and leaked prompt instructions."""
        response = """
            ANSWER:
            Local AI can keep data within local infrastructure.

            LIMITATIONS:
            The evidence does not provide exact cost figures.

            RATIONALE:
            The answer follows the supplied evidence.

            Rules:
            - Do not add unsupported facts.
            - Do not create a SOURCES section.
            """

        cleaned = clean_research_response(response)

        self.assertIn(
            "Local AI can keep data within local infrastructure.",
            cleaned
        )

        self.assertIn(
            "The evidence does not provide exact cost figures.",
            cleaned
        )

        self.assertNotIn("RATIONALE:", cleaned)

        self.assertNotIn(
            "The answer follows the supplied evidence.",
            cleaned
        )

        self.assertNotIn("Rules:", cleaned)

        self.assertNotIn(
            "Do not create a SOURCES section.",
            cleaned
        )
        
    def test_save_report_creates_report_file(self):
        """Save the question, report, and both local and web citations."""
        sources = [
            {
                "type": "local",
                "name": "ai_notes.txt",
                "url": None
            },
            {
                "type": "web",
                "name": "Example Research Page",
                "url": "https://example.com/research"
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_folder:
            report_path = save_report(
                question="What is local AI?",
                report=(
                    "ANSWER:\n"
                    "Local AI runs locally.\n\n"
                    "LIMITATIONS:\n"
                    "The evidence is limited."
                ),
                sources=sources,
                folder_path=temporary_folder
            )

            self.assertTrue(report_path.exists())

            saved_content = report_path.read_text(
                encoding="utf-8"
            )

            self.assertIn(
                "What is local AI?",
                saved_content
            )

            self.assertIn(
                "ANSWER:",
                saved_content
            )

            self.assertIn(
                "ai_notes.txt",
                saved_content
            )

            self.assertIn(
                "https://example.com/research",
                saved_content
            )

    def test_list_saved_reports_finds_report_files(self):
        """Find timestamped reports and ignore unrelated text files."""
        with tempfile.TemporaryDirectory() as temporary_folder:
            folder = Path(temporary_folder)

            first_report = (
                folder / "research_report_20260101_100000.txt"
            )

            second_report = (
                folder / "research_report_20260101_110000.txt"
            )

            ignored_file = folder / "notes.txt"

            first_report.write_text(
                "First report",
                encoding="utf-8"
            )

            second_report.write_text(
                "Second report",
                encoding="utf-8"
            )

            ignored_file.write_text(
                "Not a research report",
                encoding="utf-8"
            )

            reports = list_saved_reports(folder)

            self.assertEqual(len(reports), 2)

            report_names = {
                report.name
                for report in reports
            }

            self.assertEqual(
                report_names,
                {
                    "research_report_20260101_100000.txt",
                    "research_report_20260101_110000.txt"
                }
            )

    def test_read_saved_report_returns_saved_content(self):
        """Read a saved report and return its text without altering it."""
        with tempfile.TemporaryDirectory() as temporary_folder:
            report_path = (
                Path(temporary_folder) / "research_report_test.txt"
            )

            report_path.write_text(
                (
                    "LOCAL RESEARCH ASSISTANT\n\n"
                    "QUESTION:\n"
                    "What is local AI?\n\n"
                    "ANSWER:\n"
                    "Local AI runs locally."
                ),
                encoding="utf-8"
            )

            content = read_saved_report(report_path)

            self.assertIn(
                "What is local AI?",
                content
            )

            self.assertIn(
                "Local AI runs locally.",
                content
            )

    def test_load_config_rejects_missing_file(self):
        """Raise a configuration error when the requested file is absent."""
        with tempfile.TemporaryDirectory() as temporary_folder:
            missing_path = (
                Path(temporary_folder) / "missing_config.json"
            )

            with self.assertRaises(RuntimeError):
                load_config(missing_path)

    def test_load_config_rejects_invalid_json(self):
        """Raise a configuration error when the file is not valid JSON."""
        with tempfile.TemporaryDirectory() as temporary_folder:
            config_path = (
                Path(temporary_folder) / "config.json"
            )

            config_path.write_text(
                "{invalid json",
                encoding="utf-8"
            )

            with self.assertRaises(RuntimeError):
                load_config(config_path)

    def test_run_research_is_available(self):
        """Confirm that the complete research workflow is callable."""
        self.assertTrue(callable(run_research))

    def test_clean_research_response_removes_extra_sections(self):
        """Keep the required report sections and remove extra model sections."""
        response = """
    ANSWER:
    Joule Agents automate business workflows.

    Answer:
    This is a duplicated answer.

    LIMITATIONS:
    The evidence does not describe every implementation detail.

    RATIONALE:
    This explanation should not appear.

    SOURCES:
    The model should not provide sources.
    """

        cleaned = clean_research_response(response)

        self.assertIn(
            "Joule Agents automate business workflows.",
            cleaned
        )

        self.assertIn(
            "The evidence does not describe every implementation detail.",
            cleaned
        )

        self.assertNotIn(
            "This is a duplicated answer.",
            cleaned
        )

        self.assertNotIn("RATIONALE:", cleaned)
        self.assertNotIn(
            "The model should not provide sources.",
            cleaned
        )

if __name__ == "__main__":
    unittest.main()