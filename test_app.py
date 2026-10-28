import unittest
from unittest.mock import patch, MagicMock
from urllib.error import URLError, HTTPError

from app import ask_ollama

#Command to test python -m unittest test_app.py
#python -m unittest discover
#unittest discover automatically finds files named like: test*.py and runs the tests in them.

class TestLocalResearchAssistant(unittest.TestCase):

    def test_ask_ollama_is_available(self):
        self.assertTrue(callable(ask_ollama)) #checks that ask_ollama exists and can be called as a function.

    @patch("app.urlopen")
    def test_ask_ollama_returns_response(self, mock_urlopen):
        fake_response = MagicMock()
        fake_response.read.return_value = (
            b'{"response": "This is a test response."}'
        )

        mock_urlopen.return_value.__enter__.return_value = fake_response

        result = ask_ollama("Test question")

        self.assertEqual(result, "This is a test response.")    

    @patch("app.urlopen")
    def test_ask_ollama_handles_connection_error(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("Connection refused") #When the application calls urlopen(), raise this error instead of making a network request.

        result = ask_ollama("Test question")

        self.assertEqual(
            result,
            "Could not connect to Ollama: Connection refused"
        )                

    @patch("app.urlopen")
    def test_ask_ollama_handles_missing_response_field(self, mock_urlopen):
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

if __name__ == "__main__":
    unittest.main()