# Local Research Assistant

A local AI research application built with Python, Ollama, Tavily, and Tkinter.

This project was created as a hands-on introduction to AI engineering. It demonstrates what actually happens when an application “does research”:

- It gathers evidence from local documents and the web.
- It converts questions and evidence into embeddings.
- It retrieves and ranks the most relevant evidence.
- It asks a local language model to generate an evidence-grounded answer.
- It preserves local filenames and web URLs.
- It reports limitations.
- It saves reports for later retrieval.
- It provides both a command-line interface and a desktop GUI.

This is not a basic chatbot. It is a small Retrieval-Augmented Generation, or RAG, application.

---

## 1. Project Objective

The Local Research Assistant accepts a research question and uses two evidence sources:

1. Local `.txt` documents supplied by the user.
2. Current web-search results returned by Tavily.

The application ranks the available evidence, sends only the strongest evidence to a local language model running through Ollama, generates a structured research report, displays the result through a CLI or Tkinter desktop UI, and saves the report for later retrieval.

Example research question:

```text
Should a small business use local AI models or cloud-based AI?
Compare costs, privacy, hardware requirements, and limitations.
Provide evidence and a recommendation.
```

---

## 2. Main Capabilities

- Accept user-defined research questions.
- Read multiple local `.txt` documents.
- Search the live web using Tavily.
- Preserve local filenames and web URLs.
- Split documents into contextual, overlapping chunks.
- Generate local embeddings using Ollama and `nomic-embed-text`.
- Rank evidence using semantic similarity and keyword overlap.
- Reject evidence below a minimum similarity threshold.
- Combine local-document and web evidence.
- Generate evidence-grounded answers using `qwen3:0.6b`.
- Produce structured `ANSWER`, `LIMITATIONS`, and `SOURCES` sections.
- Prevent the language model from becoming the authority for source attribution.
- Save reports as timestamped text files.
- List and read previously saved reports.
- Run from PowerShell or a Tkinter desktop interface.
- Validate behavior using automated unit tests.

---

## 3. Technology Stack

### Programming language

- Python 3

### Local AI runtime

- Ollama

Ollama runs the language and embedding models locally and exposes HTTP APIs that Python calls.

### Generation model

- `qwen3:0.6b`

This model receives the selected evidence and generates the answer and limitations.

### Embedding model

- `nomic-embed-text`

This model converts questions and evidence chunks into numerical vectors for semantic retrieval.

### Web research

- Tavily Search API

Tavily returns current web-search results containing:

- page title;
- URL;
- extracted page content.

### User interface

- Tkinter
- `tkinter.scrolledtext`

### Storage

- Local `.txt` files for research evidence.
- Local `.txt` files for saved reports.
- `config.json` for local configuration.

### Testing

- Python `unittest`
- `unittest.mock`
- `tempfile`
- `pathlib.Path`

### Python standard-library modules used

- `json`
- `urllib.request`
- `urllib.error`
- `re`
- `pathlib`
- `math`
- `datetime`
- `tkinter`
- `unittest`
- `tempfile`

No large AI framework such as LangChain is required for version 1. The workflow is implemented directly so each AI-engineering layer remains visible and understandable.

---

## 4. Project Structure

```text
local-research-assistant/
├── app.py
├── ui.py
├── test_app.py
├── README.md
├── .gitignore
├── config.example.json
├── config.json
├── documents/
│   ├── ai_notes.txt
│   └── cloud_ai_notes.txt
└── reports/
    └── research_report_YYYYMMDD_HHMMSS.txt
```

### File responsibilities

#### `app.py`

Contains:

- Ollama generation integration;
- Ollama embedding integration;
- local-document loading;
- web-search integration;
- chunking;
- semantic and keyword retrieval;
- prompt construction;
- output cleaning;
- source attribution;
- report generation;
- report storage;
- CLI execution;
- reusable `run_research()` workflow.

#### `ui.py`

Contains the Tkinter desktop user interface.

It lets the user:

- enter a research question;
- start research;
- view the generated report;
- view source URLs;
- see the saved report path;
- browse previously saved reports.

#### `test_app.py`

Contains automated tests for:

- Ollama responses;
- network failures;
- invalid JSON;
- prompt construction;
- document loading;
- chunking;
- keyword retrieval;
- semantic retrieval;
- similarity filtering;
- source provenance;
- response cleanup;
- report saving;
- report listing;
- report reading;
- configuration validation.

#### `config.json`

Contains the real Tavily API key.

This file must remain local and must not be committed to source control.

#### `config.example.json`

Provides a configuration template without exposing a real API key.

#### `documents/`

Contains local `.txt` files used as research evidence.

#### `reports/`

Contains timestamped reports generated by the application.

#### `.gitignore`

Prevents secrets, generated reports, and Python cache files from being committed.

---

## 5. High-Level Architecture

```text
User
  |
  v
CLI or Tkinter UI
  |
  v
run_research(question)
  |
  +-----------------------------+
  |                             |
  v                             v
Local .txt documents       Tavily web search
  |                             |
  v                             v
Contextual local chunks    Web evidence chunks
  |                             |
  +-------------+---------------+
                |
                v
       Combined evidence pool
                |
                v
     Ollama embedding model
       nomic-embed-text
                |
                v
 Semantic similarity + keyword overlap
                |
                v
 Minimum similarity threshold + top-k selection
                |
                v
     Selected supporting evidence
                |
                v
       Evidence-grounded prompt
                |
                v
 Local Ollama generation model
          qwen3:0.6b
                |
                v
 Python output cleaning and source control
                |
                v
 ANSWER + LIMITATIONS + SOURCES
                |
                v
 Timestamped saved report
```

### Architectural responsibilities

- Python controls the complete application workflow.
- Ollama hosts and runs the local AI models.
- Tavily supplies current web evidence.
- The embedding model supports retrieval.
- The generation model writes the answer and limitations.
- Python controls source attribution and URLs.
- The CLI and UI both call the same reusable research engine.
- Reports are stored locally for later retrieval.

---

## 6. Program Execution Flow

### Step 1: Start the application

The user starts either the CLI:

```powershell
python app.py
```

or the desktop UI:

```powershell
python ui.py
```

### Step 2: Enter a research question

The application accepts the question and removes accidental spaces from the beginning and end.

An empty question is rejected.

### Step 3: Load configuration

`load_config()` reads `config.json`.

It validates that the file:

- exists;
- contains valid JSON;
- contains a JSON object;
- includes the Tavily API key required for web research.

### Step 4: Load local documents

`load_text_documents()` reads every `.txt` file under the `documents/` folder.

Each document is represented as:

```python
{
    "source": "ai_notes.txt",
    "content": "Document content..."
}
```

### Step 5: Create contextual chunks

`create_contextual_sourced_chunks()` splits documents into paragraphs and creates overlapping windows.

With a window size of two:

```text
Chunk 1 = paragraph 1 + paragraph 2
Chunk 2 = paragraph 2 + paragraph 3
Chunk 3 = paragraph 3 + paragraph 4
```

Overlapping chunks retain context that might otherwise be lost when paragraphs are separated.

Each local chunk contains:

```python
{
    "source": "ai_notes.txt",
    "content": "Combined paragraph content...",
    "source_type": "local",
    "url": None
}
```

### Step 6: Search the web

`search_web()` sends the research question to Tavily.

Tavily returns results containing:

- title;
- URL;
- extracted content.

### Step 7: Convert web results into evidence chunks

`create_web_chunks()` transforms Tavily results into the same general structure used by local evidence.

Example:

```python
{
    "source": "Web page title",
    "url": "https://example.com/page",
    "content": "Extracted web content...",
    "source_type": "web"
}
```

Results without content are skipped.

### Step 8: Combine local and web evidence

Python combines:

```text
local contextual chunks
+
web evidence chunks
=
combined evidence pool
```

### Step 9: Embed the research question

`get_embedding()` sends the question to the Ollama embedding API.

The question is prefixed with:

```text
search_query:
```

This tells the embedding model that the text is a search query.

### Step 10: Embed each evidence chunk

Each evidence chunk is sent to the same embedding model with:

```text
search_document:
```

This tells the embedding model that the text is searchable evidence.

### Step 11: Calculate semantic similarity

`cosine_similarity()` compares the question vector with each evidence vector.

A higher score means the question and chunk are more closely related in meaning.

### Step 12: Extract normalized keywords

`extract_keywords()`:

1. converts text to lowercase;
2. extracts words;
3. removes common stop words;
4. applies term aliases;
5. returns normalized keywords.

Example aliases:

```text
benefits       -> advantages
confidential   -> privacy
business       -> organization
infrastructure -> hardware
```

### Step 13: Apply hybrid retrieval

`retrieve_semantic_chunks()` combines:

- semantic similarity from embeddings;
- normalized keyword overlap.

The current default weighting is:

```text
60% semantic similarity
40% keyword overlap
```

The semantic score is also used as a safety filter.

Evidence must pass the minimum semantic-similarity threshold before keyword overlap can improve its ranking.

### Step 14: Select the strongest evidence

The application sorts all acceptable chunks by hybrid score and keeps only the strongest top-k results.

When local and web evidence are combined, the main workflow currently requests up to five evidence chunks.

### Step 15: Handle insufficient evidence

If no evidence passes the minimum similarity threshold:

- Python creates an insufficient-evidence report.
- Ollama is not called.
- No unrelated source is presented as supporting evidence.

### Step 16: Build the research prompt

`build_research_prompt()` combines:

- selected evidence;
- the research question;
- evidence-grounding instructions;
- required output sections.

The model is instructed to produce:

```text
ANSWER:
...

LIMITATIONS:
...
```

It is told not to invent unsupported:

- facts;
- mechanisms;
- safeguards;
- security controls;
- recommendations;
- sources.

### Step 17: Generate the answer

`ask_ollama()` sends the prompt to the local Ollama `/api/generate` endpoint.

Ollama runs:

```text
qwen3:0.6b
```

and returns the generated response to Python.

### Step 18: Clean the generated response

`clean_research_response()` enforces the report structure.

It removes:

- repeated answers;
- duplicated headings;
- prompt leakage;
- unexpected rationale sections;
- model-generated rules;
- model-generated sources;
- unsupported extra sections.

Python retains one:

```text
ANSWER
```

and one:

```text
LIMITATIONS
```

section.

### Step 19: Build authoritative source references

`get_source_references()` creates the source list directly from the chunks selected during retrieval.

Local sources include:

```text
filename
```

Web sources include:

```text
page title
URL
```

Python creates the source list because Python knows exactly which evidence was retrieved.

The language model is not trusted to invent or reproduce sources.

### Step 20: Save the report

`save_report()` saves the question, report, and source references.

Example filename:

```text
reports/research_report_20261028_120545.txt
```

### Step 21: Return the structured result

`run_research()` returns:

```python
{
    "question": question,
    "report": report,
    "sources": source_references,
    "relevant_chunks": relevant_chunks,
    "report_path": report_path
}
```

Both the CLI and Tkinter UI display this result.

---

## 7. AI Concepts Used

### 7.1 Large Language Model

A Large Language Model, or LLM, generates text from a prompt.

In this project:

```text
qwen3:0.6b
```

generates the final answer and limitations.

Important lesson:

The LLM does not automatically search the web or read files. Python must explicitly provide the evidence.

### 7.2 Prompt engineering

Prompt engineering means creating instructions and context that give the language model a clear task.

The research prompt contains:

- the model’s role;
- retrieved evidence;
- the research question;
- required output sections;
- grounding rules;
- limitations instructions.

### 7.3 Embeddings

An embedding converts text into a numerical vector.

Conceptually:

```text
"Local AI protects sensitive data"
        |
        v
[0.12, -0.44, 0.81, 0.05, ...]
```

Individual numbers are not directly meaningful to the user.

The complete vector represents characteristics of the text’s meaning.

### 7.4 Semantic search

Semantic search retrieves evidence based on meaning rather than only exact words.

Example:

```text
Question:
How can a company protect confidential information?

Evidence:
Prompts and documents can remain within the local environment.
```

The wording is different, but embeddings can recognize that the two texts are related.

### 7.5 Keyword retrieval

Keyword retrieval looks for normalized word overlap between the question and evidence.

The project began with basic keyword retrieval before advancing to semantic retrieval.

This made the progression understandable:

```text
exact word matching
->
normalized word matching
->
semantic matching
->
hybrid retrieval
```

### 7.6 Stop words

Stop words are common words that provide little retrieval value.

Examples:

```text
the
is
of
what
are
with
```

The project removes these before calculating keyword overlap.

### 7.7 Term normalization

Different words can express similar meanings.

The project uses aliases such as:

```text
benefit       -> advantages
benefits      -> advantages
private       -> privacy
privately     -> privacy
confidential  -> privacy
information   -> data
company       -> organization
business      -> organization
own           -> controlled
infrastructure -> hardware
```

This improves keyword retrieval for a controlled domain.

### 7.8 Hybrid retrieval

Hybrid retrieval combines:

- semantic similarity;
- keyword overlap.

Semantic similarity recognizes paraphrases.

Keyword overlap strengthens evidence containing important exact or normalized terms.

Current default weights:

```text
semantic_weight = 0.60
keyword_weight = 0.40
```

### 7.9 Cosine similarity

Cosine similarity compares the direction of two vectors.

The calculation uses:

- a dot product;
- the magnitude of vector A;
- the magnitude of vector B.

Conceptually:

```text
similarity =
dot_product /
(magnitude_a * magnitude_b)
```

Higher scores indicate stronger semantic similarity.

### 7.10 Chunking

Chunking divides documents into smaller searchable units.

Sending an entire collection of documents to a model would:

- consume unnecessary context;
- include irrelevant information;
- make attribution harder;
- eventually exceed context limits.

### 7.11 Contextual overlapping chunks

Paragraphs can depend on neighboring paragraphs.

The application therefore creates overlapping chunks:

```text
Paragraph 1 + Paragraph 2
Paragraph 2 + Paragraph 3
Paragraph 3 + Paragraph 4
```

This preserves surrounding context.

### 7.12 Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, means retrieving evidence before generating an answer.

This project’s RAG flow is:

```text
Question
  ->
retrieve local and web evidence
  ->
rank evidence
  ->
filter weak evidence
  ->
give selected evidence to the LLM
  ->
generate a grounded report
```

### 7.13 Grounding

Grounding constrains the language model to supplied evidence.

The model is instructed not to add unsupported facts, security measures, mechanisms, or recommendations.

### 7.14 Hallucination mitigation

The project reduces hallucination risk through:

- evidence-only prompting;
- semantic relevance filtering;
- deterministic insufficient-evidence handling;
- Python-controlled source attribution;
- response cleaning;
- limitations reporting;
- top-k retrieval.

These measures reduce hallucination risk but do not guarantee perfect accuracy.

### 7.15 Similarity threshold

Evidence below:

```python
min_similarity=0.55
```

is rejected.

The threshold was selected after comparing scores from relevant and unrelated questions in this project.

It may require recalibration for other domains or embedding models.

### 7.16 Top-k retrieval

Top-k retrieval keeps only the strongest `k` evidence chunks.

This prevents the prompt from being filled with excessive or weak evidence.

### 7.17 Source provenance

Provenance means recording where evidence came from.

The project preserves:

- local filenames;
- web page titles;
- web URLs;
- source type.

### 7.18 Tool integration

The application coordinates several tools:

- Ollama generation API;
- Ollama embedding API;
- Tavily Search API;
- local filesystem;
- Tkinter interface.

Python acts as the orchestrator.

### 7.19 Separation of concerns

Each component has one primary responsibility:

```text
Tavily
-> retrieves current web information

nomic-embed-text
-> converts text into vectors

retrieval functions
-> select evidence

qwen3:0.6b
-> generates answer and limitations

Python
-> controls workflow, structure, sources, and storage

Tkinter
-> provides the user interface
```

### 7.20 Mocking

Automated tests replace network operations with controlled fake behavior.

This is called mocking.

Mocking makes tests:

- fast;
- deterministic;
- repeatable;
- independent of Ollama availability;
- independent of Tavily availability;
- suitable for testing failure paths.

---

## 8. Function Reference

### `read_document(file_path)`

Reads a UTF-8 text file and returns its content.

```python
def read_document(file_path):
```

Input:

- path to a text file.

Output:

- string containing the file’s text.

---

### `ask_ollama(prompt)`

Sends a prompt to the local Ollama generation API.

```python
def ask_ollama(prompt):
```

Responsibilities:

- build the JSON request;
- call `/api/generate`;
- request `qwen3:0.6b`;
- disable streaming;
- parse the JSON response;
- return the generated text.

Handles:

- HTTP errors;
- connection errors;
- invalid JSON;
- missing response fields.

---

### `get_embedding(text, input_type="document")`

Converts text into an embedding vector.

```python
def get_embedding(text, input_type="document"):
```

Supported input types:

```text
query
document
```

Query text receives:

```text
search_query:
```

Document text receives:

```text
search_document:
```

Returns:

```python
[0.12, -0.44, 0.81, ...]
```

Raises a `RuntimeError` if the embedding API fails.

---

### `build_research_prompt(question, evidence)`

Builds the evidence-grounded prompt.

```python
def build_research_prompt(question, evidence):
```

The prompt contains:

- evidence;
- question;
- answer instructions;
- limitations instructions;
- grounding rules.

---

### `split_into_chunks(document)`

Splits text into non-empty paragraphs.

```python
def split_into_chunks(document):
```

Paragraphs are separated by blank lines.

---

### `create_sourced_chunks(documents)`

Creates basic paragraph-level chunks.

```python
def create_sourced_chunks(documents):
```

Each chunk retains:

- source filename;
- chunk content.

This function remains useful for testing and for understanding the earlier retrieval version.

---

### `extract_keywords(text)`

Extracts normalized keywords.

```python
def extract_keywords(text):
```

It:

1. converts text to lowercase;
2. extracts alphanumeric words;
3. removes stop words;
4. applies aliases;
5. returns a set.

Example:

```python
extract_keywords(
    "What are the benefits of running AI privately?"
)
```

may produce:

```python
{
    "advantages",
    "running",
    "ai",
    "privacy"
}
```

---

### `retrieve_relevant_chunks(question, chunks, top_k=3)`

Implements the earlier keyword-overlap retrieval method.

```python
def retrieve_relevant_chunks(
    question,
    chunks,
    top_k=3
):
```

It:

- extracts question keywords;
- extracts chunk keywords;
- calculates overlap;
- sorts by overlap;
- returns top results.

This function demonstrates the simpler retrieval method used before embeddings.

---

### `load_text_documents(folder_path)`

Loads every `.txt` file in a folder.

```python
def load_text_documents(folder_path):
```

Returns:

```python
[
    {
        "source": "ai_notes.txt",
        "content": "..."
    }
]
```

---

### `get_source_names(relevant_chunks)`

Returns unique source filenames.

```python
def get_source_names(relevant_chunks):
```

This was the earlier local-document source helper.

The newer combined local and web workflow uses `get_source_references()`.

---

### `cosine_similarity(vector_a, vector_b)`

Calculates the cosine similarity between two vectors.

```python
def cosine_similarity(
    vector_a,
    vector_b
):
```

It:

- validates equal vector dimensions;
- calculates the dot product;
- calculates vector magnitudes;
- handles zero vectors;
- returns the similarity score.

---

### `retrieve_semantic_chunks(...)`

Performs hybrid semantic retrieval.

```python
def retrieve_semantic_chunks(
    question,
    chunks,
    top_k=3,
    min_similarity=0.55,
    semantic_weight=0.60,
    keyword_weight=0.40
):
```

Responsibilities:

1. Embed the research question.
2. Embed each evidence chunk.
3. Calculate semantic similarity.
4. Reject weak semantic matches.
5. Calculate keyword overlap.
6. Calculate the hybrid score.
7. Sort chunks by hybrid score.
8. Return the top-k results.
9. Preserve source metadata and URLs.

Hybrid formula:

```text
hybrid_score =
semantic_weight * semantic_score
+
keyword_weight * keyword_score
```

---

### `create_contextual_sourced_chunks(documents, window_size=2)`

Creates overlapping local-document chunks.

```python
def create_contextual_sourced_chunks(
    documents,
    window_size=2
):
```

Each result contains:

```python
{
    "source": "ai_notes.txt",
    "content": "...",
    "source_type": "local",
    "url": None
}
```

Raises `ValueError` if the window size is below one.

---

### `load_config(file_path="config.json")`

Loads and validates configuration.

```python
def load_config(
    file_path="config.json"
):
```

Checks:

- whether the file exists;
- whether the JSON is valid;
- whether the root value is a JSON object.

Returns a Python dictionary.

---

### `search_web(query, api_key, max_results=3)`

Searches the web through Tavily.

```python
def search_web(
    query,
    api_key,
    max_results=3
):
```

Responsibilities:

- validate the API key;
- construct the Tavily request;
- send the HTTP request;
- parse the response;
- return search results.

---

### `create_web_chunks(web_results)`

Converts Tavily results into evidence chunks.

```python
def create_web_chunks(web_results):
```

Each web chunk contains:

```python
{
    "source": "Page title",
    "url": "https://example.com",
    "content": "Extracted page content",
    "source_type": "web"
}
```

Results without content are skipped.

---

### `get_source_references(relevant_chunks)`

Creates unique structured source references.

```python
def get_source_references(
    relevant_chunks
):
```

Local source:

```python
{
    "type": "local",
    "name": "ai_notes.txt",
    "url": None
}
```

Web source:

```python
{
    "type": "web",
    "name": "Web page title",
    "url": "https://example.com"
}
```

Duplicate references are removed.

---

### `clean_research_response(response)`

Enforces the final report structure.

```python
def clean_research_response(response):
```

It keeps one:

```text
ANSWER:
```

and one:

```text
LIMITATIONS:
```

It removes unexpected model-generated sections such as:

```text
Answer:
RATIONALE:
Rules:
SOURCES:
RECOMMENDATION:
```

---

### `save_report(question, report, sources, folder_path="reports")`

Saves a completed report.

```python
def save_report(
    question,
    report,
    sources,
    folder_path="reports"
):
```

It:

- creates the reports folder if necessary;
- generates a timestamped filename;
- writes the question;
- writes the answer and limitations;
- writes local and web sources;
- returns the saved file path.

---

### `list_saved_reports(folder_path="reports")`

Lists saved reports.

```python
def list_saved_reports(
    folder_path="reports"
):
```

It:

- checks whether the folder exists;
- finds matching `research_report_*.txt` files;
- sorts them newest first;
- returns `Path` objects.

---

### `read_saved_report(report_path)`

Reads a saved report.

```python
def read_saved_report(report_path):
```

Raises `FileNotFoundError` if the report does not exist.

---

### `run_research(...)`

Runs the complete reusable research workflow.

```python
def run_research(
    question,
    config_path="config.json",
    documents_folder="documents",
    reports_folder="reports"
):
```

Responsibilities:

1. Validate the question.
2. Load configuration.
3. Load local documents.
4. Create contextual local chunks.
5. Search Tavily.
6. Create web chunks.
7. Combine local and web evidence.
8. Rank evidence.
9. Handle insufficient evidence.
10. Build the research prompt.
11. Call Ollama.
12. Clean the answer.
13. create source references.
14. Save the report.
15. Return structured results.

Returns:

```python
{
    "question": question,
    "report": report,
    "sources": source_references,
    "relevant_chunks": relevant_chunks,
    "report_path": report_path
}
```

---

### UI function: `run_research_from_ui()`

This function:

- gets the question from the UI;
- validates the question;
- disables the Research button during execution;
- displays a status message;
- calls `run_research()`;
- displays the report;
- displays sources and URLs;
- displays the saved report path;
- catches and displays errors;
- enables the button when finished.

---

### UI function: `show_saved_reports()`

This function:

- finds existing reports;
- opens a second window;
- shows report filenames;
- allows one report to be selected;
- reads the selected report;
- displays its content.

---

## 9. Setup Instructions

### 9.1 Verify Python

```powershell
python --version
```

### 9.2 Verify Ollama

```powershell
ollama --version
```

### 9.3 Install the generation model

```powershell
ollama pull qwen3:0.6b
```

### 9.4 Install the embedding model

```powershell
ollama pull nomic-embed-text
```

### 9.5 Verify installed models

```powershell
ollama list
```

Expected model names include:

```text
qwen3:0.6b
nomic-embed-text
```

### 9.6 Verify the Ollama API

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

### 9.7 Configure Tavily

Copy:

```text
config.example.json
```

to:

```text
config.json
```

Add the actual Tavily API key:

```json
{
    "tavily_api_key": "YOUR_REAL_TAVILY_API_KEY"
}
```

Do not share or commit the real key.

### 9.8 Add local evidence documents

Place `.txt` files under:

```text
documents/
```

Separate paragraphs with blank lines because blank lines are used as chunk boundaries.

---

## 10. Commands Needed to Execute the Application

### Run the CLI

```powershell
python app.py
```

### Run the desktop UI

```powershell
python ui.py
```

### Verify Tkinter

```powershell
python -m tkinter
```

### Run the full automated test suite

```powershell
python -m unittest test_app.py
```

### Automatically discover every `test*.py` file

```powershell
python -m unittest discover
```

### Validate Python syntax

```powershell
python -m py_compile app.py ui.py test_app.py
```

### Verify `run_research()` can be imported

```powershell
python -c "from app import run_research; print('run_research import OK')"
```

### Verify Ollama models

```powershell
ollama list
```

### Start Ollama if required

```powershell
ollama serve
```

If Ollama is already running as a Windows application or service, this separate command may not be needed.

---

## 11. Configuration and Security

### `config.json`

Contains the real Tavily API key.

Example:

```json
{
    "tavily_api_key": "YOUR_REAL_TAVILY_API_KEY"
}
```

This file must remain local.

### `config.example.json`

Safe template:

```json
{
    "tavily_api_key": "YOUR_TAVILY_API_KEY"
}
```

### `.gitignore`

Recommended contents:

```gitignore
# Local secrets
config.json

# Python generated files
__pycache__/
*.pyc

# Generated reports
reports/

# Virtual environments
.venv/
venv/
```

### Office-laptop consideration

This project uses a local JSON configuration because persistent environment-variable changes may be restricted on managed corporate devices.

The key remains sensitive even when stored locally.

Do not:

- commit `config.json`;
- send it through email or chat;
- paste the real key into documentation;
- place the real key inside `app.py`;
- include it in screenshots.

---

## 12. Testing Strategy

The automated tests cover:

- availability of the Ollama function;
- successful Ollama responses;
- Ollama connection errors;
- Ollama HTTP errors;
- invalid Ollama JSON;
- missing Ollama response fields;
- prompt construction;
- stop-word removal;
- alias normalization;
- keyword ranking;
- top-k limits;
- local document loading;
- source-aware chunking;
- multi-document retrieval;
- source de-duplication;
- semantic retrieval;
- similarity-threshold filtering;
- contextual overlapping chunks;
- web chunk creation;
- preservation of web URLs;
- response cleanup;
- report saving;
- saved-report discovery;
- saved-report reading;
- missing configuration handling;
- invalid configuration JSON;
- availability of `run_research()`.

### Why network calls are mocked

Real network calls would make automated tests:

- slower;
- dependent on internet access;
- dependent on Ollama;
- dependent on Tavily;
- harder to reproduce;
- potentially subject to API usage limits.

The tests therefore use `unittest.mock.patch()` to replace functions such as:

```python
urlopen()
get_embedding()
```

with controlled behavior.

### Manual acceptance testing

The project also includes manual testing for:

- CLI research;
- GUI research;
- unrelated questions;
- insufficient evidence;
- combined local and web evidence;
- source URL display;
- report saving;
- saved report browsing.

---

## 13. Error-Handling Design

### Empty question

The application rejects blank questions.

### Missing `.txt` documents

The application raises a readable error if the `documents/` folder does not contain research documents.

### Missing `config.json`

`load_config()` raises:

```text
Configuration file not found
```

### Invalid `config.json`

The application raises a readable error instead of returning a long JSON traceback.

### Missing Tavily key

The application reports that the Tavily API key is missing.

### Ollama generation error

`ask_ollama()` reports:

- HTTP failure;
- connection failure;
- unexpected response data.

### Ollama embedding error

`get_embedding()` raises `RuntimeError`.

This is intentional because retrieval expects a numerical vector. An error string must not accidentally enter mathematical calculations.

### Tavily error

`search_web()` reports:

- missing key;
- HTTP failure;
- connection failure;
- invalid response.

### No relevant evidence

Python creates a deterministic insufficient-evidence report.

Ollama is not called.

No unrelated sources are presented as supporting evidence.

### Missing saved report

`read_saved_report()` raises `FileNotFoundError`.

---

## 14. Important Engineering Lessons

### The model is not the application

Ollama runs models, but Python controls the workflow.

### A language model does not automatically research

Web search and document ingestion must be implemented as separate capabilities.

### Retrieval and generation are different tasks

The embedding model retrieves evidence.

The generation model writes the answer.

### Source attribution should be deterministic

Python knows which chunks were selected.

Python therefore owns the source list.

### A successful path is not enough

A real application must handle:

- missing files;
- invalid configuration;
- connection failures;
- HTTP failures;
- malformed responses;
- irrelevant evidence;
- empty questions.

### Tests protect incremental development

The test suite caught real issues including:

- missing imports;
- incorrect indentation;
- lost chunks;
- incorrectly scoped functions;
- malformed configuration handling;
- source URL loss;
- prompt leakage;
- duplicate AI sections.

### Refactoring enables reuse

Extracting:

```python
run_research()
```

allowed both the CLI and GUI to use the same research engine.

### Retrieval quality must be observed

The semantic threshold and hybrid weights were selected after inspecting actual similarity scores for relevant and unrelated questions.

They are project settings, not universal constants.

### Small models need application-level controls

`qwen3:0.6b` sometimes repeated sections or leaked instructions.

Python therefore cleans and validates the final structure.

---

## 15. Known Limitations

- Only local `.txt` documents are supported.
- PDF and DOCX extraction are not included.
- Chunking is paragraph-based rather than token-aware.
- Chunk embeddings are recalculated during each research run.
- There is no embedding cache.
- There is no vector database.
- Tavily can return secondary or low-quality web sources.
- The application does not rate source authority.
- Claim-level citations are not inserted directly into answer sentences.
- `qwen3:0.6b` is a small model and may still make generation mistakes.
- Response cleanup uses regular expressions.
- Reports are text files rather than database records.
- The UI runs the research function synchronously and may temporarily appear unresponsive.
- The Tavily key is stored in plain text in `config.json`.
- Similarity thresholds may need adjustment for other domains.
- Keyword aliases are manually maintained.
- Web content extraction depends on Tavily’s returned content.

---

## 16. Suggested Version 2 Improvements

Potential improvements for a future version:

- PDF ingestion.
- DOCX ingestion.
- Token-aware chunking.
- Embedding caching.
- Vector database integration.
- FAISS, Chroma, or Qdrant.
- Source-authority scoring.
- Domain allowlists.
- Domain blocklists.
- Claim-level citations.
- PDF report export.
- DOCX report export.
- SQLite report metadata.
- Background UI execution.
- Progress indicators.
- Configurable Ollama model names.
- Configurable thresholds and weights.
- Improved web-content cleaning.
- Application logging.
- Better observability.
- Richer desktop or web UI.
- Packaging as a Windows executable.

These belong in version 2. Version 1 should remain stable.

---

## 17. Quick Revision Checklist

Use this section when revising the project later.

### What starts the application?

```text
app.py -> CLI
ui.py  -> desktop UI
```

### What controls the complete workflow?

```text
run_research()
```

### What generates the final answer?

```text
qwen3:0.6b through Ollama
```

### What creates embeddings?

```text
nomic-embed-text through Ollama
```

### What provides current web evidence?

```text
Tavily Search API
```

### What is RAG?

```text
Retrieve evidence first, then generate an answer from that evidence.
```

### What ranks evidence?

```text
Semantic similarity + normalized keyword overlap
```

### What rejects weak evidence?

```text
Minimum semantic-similarity threshold
```

### What keeps the prompt focused?

```text
Top-k retrieval
```

### Who controls sources?

```text
Python, not the language model
```

### Where are local evidence documents stored?

```text
documents/
```

### Where are generated reports stored?

```text
reports/
```

### Where is the Tavily key stored?

```text
config.json
```

### Which file must never be committed?

```text
config.json
```

### How are tests run?

```powershell
python -m unittest test_app.py
```

### How is the UI started?

```powershell
python ui.py
```

### How is the CLI started?

```powershell
python app.py
```

---

## 18. Example End-to-End Flow

Research question:

```text
Should a small business use local AI or cloud AI?
Compare privacy, hardware, costs, and limitations.
```

Execution:

```text
1. Python validates the question.

2. Python loads config.json.

3. Python loads local .txt documents.

4. Python creates overlapping local chunks.

5. Python sends the question to Tavily.

6. Tavily returns web titles, URLs, and content.

7. Python converts web results into evidence chunks.

8. Python combines local and web evidence.

9. Ollama embeds the question.

10. Ollama embeds every evidence chunk.

11. Python calculates semantic similarity.

12. Python calculates normalized keyword overlap.

13. Python creates a hybrid score.

14. Python removes evidence below the threshold.

15. Python selects the strongest top-k evidence.

16. Python builds an evidence-grounded prompt.

17. qwen3:0.6b generates an answer and limitations.

18. Python removes duplicated or unexpected sections.

19. Python builds the real source list from retrieved chunks.

20. Python saves the completed report.

21. The CLI or UI displays the result.

22. The user can reopen the saved report later.
```

---

## 19. Final Project Status

**Local Research Assistant v1: Complete**

The project demonstrates the complete beginner-friendly AI engineering lifecycle:

```text
problem definition
->
local model integration
->
HTTP API communication
->
document ingestion
->
chunking
->
embeddings
->
semantic search
->
hybrid retrieval
->
RAG
->
live web search
->
grounding
->
source provenance
->
report generation
->
persistence
->
automated testing
->
refactoring
->
CLI
->
desktop UI
->
documentation
```

The project is intentionally implemented using visible Python building blocks so each AI-engineering concept can be understood and revised later.