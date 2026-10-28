import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import re
from pathlib import Path
import math
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:0.6b"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"
TAVILY_URL = "https://api.tavily.com/search"

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "the", "to", "what", "when", "where", "which",
    "who", "why", "with"
}

# Normalize common wording before keyword overlap is scored, so equivalent
# questions (for example, "benefits" and "advantages") can match the same text.
TERM_ALIASES = {
    "benefit": "advantages",
    "benefits": "advantages",
    "advantage": "advantages",
    "private": "privacy",
    "privately": "privacy",
    "confidential": "privacy",
    "information": "data",
    "company": "organization",
    "business": "organization",
    "own": "controlled",
    "infrastructure": "hardware",
}

def read_document(file_path):
    """Read a text document and return its contents."""

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def ask_ollama(prompt):
    """Send a prompt to Ollama and return the model's response."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["response"]

    except HTTPError as error:
        return f"Ollama API error: HTTP {error.code}"

    except URLError as error:
        return f"Could not connect to Ollama: {error.reason}"

    except (KeyError, json.JSONDecodeError):
        return "Error: Ollama returned an unexpected response."


def get_embedding(text, input_type="document"):
    """Convert query or document text into an embedding vector."""

    # Nomic uses task prefixes to distinguish search queries from searchable text.
    if input_type == "query":
        prepared_text = f"search_query: {text}"
    elif input_type == "document":
        prepared_text = f"search_document: {text}"
    else:
        raise ValueError(
            "input_type must be either 'query' or 'document'."
        )

    payload = {
        "model": EMBEDDING_MODEL,
        "input": prepared_text
    }

    request = Request(
        OLLAMA_EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["embeddings"][0]

    except HTTPError as error:
        raise RuntimeError(
            f"Ollama embedding API error: HTTP {error.code}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama embedding API: {error.reason}"
        ) from error

    except (KeyError, IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Ollama returned an unexpected embedding response."
        ) from error

def build_research_prompt(question, evidence):
    """Build a strictly evidence-grounded research prompt."""

    return f"""
        You are a research assistant.

        Answer the question using only the evidence provided below.

        EVIDENCE:
        {evidence}

        QUESTION:
        {question}

        Return exactly these two sections:

        ANSWER:
        Give a concise answer supported directly by the evidence.

        LIMITATIONS:
        State what the evidence does not establish.

        Rules:
        - Do not add facts, mechanisms, recommendations, or assumptions
        that are not explicitly supported by the evidence.
        - Do not claim that encryption, access controls, compliance,
        monitoring, security protocols, or other safeguards exist unless
        the evidence explicitly mentions them.
        - If the evidence supports only a partial answer, clearly say so.
        - Do not create a SOURCES section.
        - Do not repeat these instructions in the answer.
        """

def split_into_chunks(document):
    """Split a document into non-empty paragraphs."""

    paragraphs = document.split("\n\n")

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

def create_sourced_chunks(documents):
    """Split documents into chunks while retaining source filenames."""

    sourced_chunks = []

    for document in documents:
        chunks = split_into_chunks(document["content"])

        for chunk in chunks:
            sourced_chunks.append({
                "source": document["source"],
                "content": chunk
            })

    return sourced_chunks

def extract_keywords(text):
    """Convert text into normalized, meaningful lowercase keywords."""

    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return {
        TERM_ALIASES.get(word, word)
        for word in words
        if word not in STOP_WORDS
    }


def retrieve_relevant_chunks(question, chunks, top_k=3):
    """Return the highest-ranked sourced chunks."""

    question_words = extract_keywords(question)
    scored_chunks = []

    for chunk in chunks:
        chunk_words = extract_keywords(chunk["content"])
        score = len(question_words & chunk_words)

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in scored_chunks[:top_k]
    ]

def load_text_documents(folder_path):
    """Load every .txt document from a folder."""

    folder = Path(folder_path)
    documents = []

    for file_path in sorted(folder.glob("*.txt")):
        content = read_document(file_path)

        documents.append({
            "source": file_path.name,
            "content": content
        })

    return documents


def get_source_names(relevant_chunks):
    """Return unique source filenames from retrieved chunks."""

    return sorted({
        chunk["source"]
        for chunk in relevant_chunks
    })


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two embedding vectors."""

    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding vectors must have the same dimensions.")

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    # A zero vector has no direction, so it cannot produce a cosine score.
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

def retrieve_semantic_chunks(
    question,
    chunks,
    top_k=3,
    min_similarity=0.55,
    semantic_weight=0.60,
    keyword_weight=0.40
):
    """Rank chunks using semantic similarity and keyword overlap."""

    question_embedding = get_embedding(
        question,
        input_type="query"
    )

    question_keywords = extract_keywords(question)
    scored_chunks = []

    for chunk in chunks:
        chunk_embedding = get_embedding(
            chunk["content"],
            input_type="document"
        )

        semantic_score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        # Exclude weak semantic matches before keyword overlap can boost them.
        if semantic_score < min_similarity:
            continue

        chunk_keywords = extract_keywords(
            chunk["content"]
        )

        if question_keywords:
            keyword_score = (
                len(question_keywords & chunk_keywords)
                / len(question_keywords)
            )
        else:
            keyword_score = 0.0

        # Semantic similarity finds paraphrases; keyword overlap favors exact terms.
        hybrid_score = (
            semantic_weight * semantic_score
            + keyword_weight * keyword_score
        )

        scored_chunks.append({
            "source": chunk["source"],
            "content": chunk["content"],
            "source_type": chunk.get("source_type", "local"),
            "url": chunk.get("url"),
            "similarity": semantic_score,
            "keyword_score": keyword_score,
            "hybrid_score": hybrid_score
        })

    scored_chunks.sort(
        key=lambda chunk: chunk["hybrid_score"],
        reverse=True
    )

    return scored_chunks[:top_k]

def create_contextual_sourced_chunks(
    documents,
    window_size=2
):
    """Create overlapping chunks while retaining source metadata."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    sourced_chunks = []

    for document in documents:
        paragraphs = split_into_chunks(
            document["content"]
        )

        for start_index in range(len(paragraphs)):
            window = paragraphs[
                start_index:start_index + window_size
            ]

            sourced_chunks.append({
                "source": document["source"],
                "content": "\n\n".join(window),
                "source_type": "local",
                "url": None
            })

    return sourced_chunks


def load_config(file_path="config.json"):
    """Load application configuration from a local JSON file."""

    path = Path(file_path)

    if not path.exists():
        raise RuntimeError(
            f"Configuration file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "config.json contains invalid JSON."
        ) from error

    if not isinstance(config, dict):
        raise RuntimeError(
            "config.json must contain a JSON object."
        )

    return config

def search_web(query, api_key, max_results=3):
    """Search the web with Tavily and return structured results."""

    if not api_key:
        raise RuntimeError(
            "Tavily API key is missing from config.json."
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic"
    }

    request = Request(
        TAVILY_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        return result["results"]

    except HTTPError as error:
        raise RuntimeError(
            f"Tavily API error: HTTP {error.code}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Could not connect to Tavily: {error.reason}"
        ) from error

    except (KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Tavily returned an unexpected response."
        ) from error


def create_web_chunks(web_results):
    """Convert Tavily results into source-aware evidence chunks."""

    chunks = []

    for result in web_results:
        title = result.get("title", "Untitled web result")
        url = result.get("url", "")
        content = result.get("content", "").strip()

        if not content:
            continue

        chunks.append({
            "source": title,
            "url": url,
            "content": content,
            "source_type": "web"
        })

    return chunks

def get_source_references(relevant_chunks):
    """Return unique source references from retrieved evidence."""

    references = []
    seen = set()

    for chunk in relevant_chunks:
        if chunk.get("source_type") == "web":
            reference = {
                "type": "web",
                "name": chunk["source"],
                "url": chunk.get("url")
            }
        else:
            reference = {
                "type": "local",
                "name": chunk["source"],
                "url": None
            }

        key = (
            reference["type"],
            reference["name"],
            reference["url"]
        )

        if key not in seen:
            seen.add(key)
            references.append(reference)

    return references

def clean_research_response(response):
    """Extract one ANSWER and one LIMITATIONS section."""

    text = response.strip()

    answer_match = re.search(
        r"(?is)\*{0,2}ANSWER:\*{0,2}\s*(.*?)"
        r"(?=\n\s*\*{0,2}LIMITATIONS:\*{0,2}|$)",
        text
    )

    limitations_match = re.search(
        r"(?is)\*{0,2}LIMITATIONS:\*{0,2}\s*(.*)",
        text
    )

    answer = (
        answer_match.group(1).strip()
        if answer_match
        else text
    )

    limitations = (
        limitations_match.group(1).strip()
        if limitations_match
        else "No additional limitations were identified."
    )

    # Remove unexpected extra sections generated by the model.
    stop_section_pattern = re.compile(
        r"(?im)^\s*(?:"
        r"ANSWER|RATIONALE|RULES|SOURCES|RECOMMENDATION"
        r")\s*:\s*"
    )

    answer_extra = stop_section_pattern.search(answer)

    if answer_extra:
        answer = answer[:answer_extra.start()].strip()

    limitations_extra = stop_section_pattern.search(limitations)

    if limitations_extra:
        limitations = limitations[
            :limitations_extra.start()
        ].strip()

    return (
        f"ANSWER:\n{answer}\n\n"
        f"LIMITATIONS:\n{limitations}"
    )

def save_report(question, report, sources, folder_path="reports"):
    """Save a completed research report as a text file."""

    folder = Path(folder_path)
    folder.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = folder / f"research_report_{timestamp}.txt"

    lines = [
        "LOCAL RESEARCH ASSISTANT",
        "=" * 50,
        "",
        "QUESTION:",
        question,
        "",
        report,
        "",
        "SOURCES:"
    ]

    if sources:
        for source in sources:
            if source["type"] == "web":
                lines.append(f"- {source['name']}")
                lines.append(f"  {source['url']}")
            else:
                lines.append(
                    f"- {source['name']} (local document)"
                )
    else:
        lines.append("- No supporting sources were retrieved.")

    file_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return file_path

def list_saved_reports(folder_path="reports"):
    """Return saved research reports from newest to oldest."""

    folder = Path(folder_path)

    if not folder.exists():
        return []

    reports = sorted(
        folder.glob("research_report_*.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    return reports

def read_saved_report(report_path):
    """Read and return a previously saved research report."""

    path = Path(report_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Saved report does not exist: {path}"
        )

    return path.read_text(encoding="utf-8")



def run_research(
    question,
    config_path="config.json",
    documents_folder="documents",
    reports_folder="reports"
):
    """Run the complete research workflow and return structured results."""

    question = question.strip()

    if not question:
        raise ValueError("Please enter a research question.")

    config = load_config(config_path)

    tavily_api_key = config.get("tavily_api_key")

    if not tavily_api_key:
        raise RuntimeError(
            "Tavily API key is missing from config.json."
        )

    documents = load_text_documents(documents_folder)

    if not documents:
        raise RuntimeError(
            "No .txt documents were found in the documents folder."
        )

    local_chunks = create_contextual_sourced_chunks(
        documents,
        window_size=2
    )

    web_results = search_web(
        question,
        tavily_api_key,
        max_results=3
    )

    web_chunks = create_web_chunks(web_results)

    all_chunks = local_chunks + web_chunks

    relevant_chunks = retrieve_semantic_chunks(
        question,
        all_chunks,
        top_k=5,
        min_similarity=0.55
    )

    if not relevant_chunks:
        report = (
            "ANSWER:\n"
            "The available evidence is insufficient "
            "to answer this question.\n\n"
            "LIMITATIONS:\n"
            "No evidence chunks met the minimum "
            "semantic-similarity threshold."
        )

        source_references = []

    else:
        source_references = get_source_references(
            relevant_chunks
        )

        evidence = "\n\n".join(
            (
                f"Source: {chunk['source']}\n"
                f"{chunk['content']}"
            )
            for chunk in relevant_chunks
        )

        research_prompt = build_research_prompt(
            question,
            evidence
        )

        answer = ask_ollama(research_prompt)

        report = clean_research_response(answer)

    report_path = save_report(
        question,
        report,
        source_references,
        folder_path=reports_folder
    )

    return {
        "question": question,
        "report": report,
        "sources": source_references,
        "relevant_chunks": relevant_chunks,
        "report_path": report_path
    }

if __name__ == "__main__":
    question = input(
        "Enter your research question: "
    ).strip()

    try:
        result = run_research(question)

        print("\nRESEARCH REPORT")
        print("=" * 50)

        print(result["report"])

        print("\nSOURCES:")

        if result["sources"]:
            for source in result["sources"]:
                if source["type"] == "web":
                    print(f"- {source['name']}")
                    print(f"  {source['url']}")
                else:
                    print(
                        f"- {source['name']} "
                        "(local document)"
                    )
        else:
            print(
                "- No supporting sources were retrieved."
            )

        print(
            f"\nReport saved to: "
            f"{result['report_path']}"
        )

    except (ValueError, RuntimeError) as error:
        print(f"\nError: {error}")    