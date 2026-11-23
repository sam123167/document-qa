from io import BytesIO

import streamlit as st
from docx import Document
from pypdf import PdfReader


st.set_page_config(
    page_title="AI Document Question Answering",
    page_icon="📄",
    layout="wide",
)

RELATIONS = {
    "is_a",
    "supports",
    "enables",
    "uses",
    "developed_by",
    "subset_of",
}

SAMPLE = """Python | is_a | Programming language
Python | supports | OOP
Python | enables | Automation
System A | uses | Python
Python | developed_by | Guido van Rossum
ML | subset_of | AI
DL | subset_of | ML
CNN | subset_of | DL
"""


def normalize(value):
    """Normalize explicit field values, not natural-language sentences."""
    return " ".join(value.split()).casefold()


def read_uploaded_document(uploaded):
    data = uploaded.getvalue()

    if len(data) > 2_000_000:
        raise ValueError("Please use a document smaller than 2 MB.")

    extension = uploaded.name.rsplit(".", 1)[-1].lower()

    if extension == "txt":
        text = data.decode("utf-8-sig")
        return [
            (f"{uploaded.name}: line {number}", line)
            for number, line in enumerate(text.splitlines(), 1)
        ]

    if extension == "pdf":
        reader = PdfReader(BytesIO(data))

        if len(reader.pages) > 30:
            raise ValueError("This demonstration accepts up to 30 PDF pages.")

        records = []

        for page_number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""

            for line_number, line in enumerate(text.splitlines(), 1):
                source = (
                    f"{uploaded.name}: page {page_number}, "
                    f"line {line_number}"
                )
                records.append((source, line))

        return records

    if extension == "docx":
        document = Document(BytesIO(data))

        return [
            (f"{uploaded.name}: paragraph {number}", paragraph.text)
            for number, paragraph in enumerate(document.paragraphs, 1)
        ]

    raise ValueError("Supported file formats: TXT, PDF and DOCX.")


def build_knowledge_base(records):
    sources = {}
    labels = {}

    for source, record in records:
        if not record.strip():
            continue

        fields = [field.strip() for field in record.split("|")]

        if len(fields) != 3 or not all(fields):
            raise ValueError(
                f"{source}: expected subject | predicate | object."
            )

        subject, predicate, obj = map(normalize, fields)

        if predicate not in RELATIONS:
            raise ValueError(
                f"{source}: unsupported predicate '{fields[1]}'."
            )

        fact = (subject, predicate, obj)
        sources.setdefault(fact, set()).add(source)

        labels.setdefault(subject, fields[0])
        labels.setdefault(obj, fields[2])

    if not sources:
        raise ValueError(
            "No usable records found. Use a structured text document. "
            "Scanned PDFs are not supported."
        )

    if len(sources) > 200:
        raise ValueError("This demonstration accepts up to 200 direct facts.")

    facts = set(sources)
    direct_count = len(facts)
    parents = {}

    subset_entities = {
        entity
        for subject, predicate, obj in facts
        if predicate == "subset_of"
        for entity in (subject, obj)
    }

    if len(subset_entities) > 30:
        raise ValueError(
            "Use at most 30 entities in subset relationships for this demo."
        )

    # Apply subset transitivity repeatedly until no new facts appear.
    while True:
        edges = sorted(
            (subject, obj)
            for subject, predicate, obj in facts
            if predicate == "subset_of"
        )

        additional = {}

        for a, b in edges:
            for x, c in edges:
                if b != x:
                    continue

                new_fact = (a, "subset_of", c)

                if new_fact not in facts:
                    additional.setdefault(
                        new_fact,
                        (
                            (a, "subset_of", b),
                            (x, "subset_of", c),
                        ),
                    )

        if not additional:
            break

        for fact, (left, right) in additional.items():
            sources[fact] = sources[left] | sources[right]
            parents[fact] = (left, right)

        facts.update(additional)

    index = {}

    for subject, predicate, obj in facts:
        index.setdefault((subject, predicate), set()).add(obj)

    return facts, index, sources, parents, labels, direct_count


def ask(question, index):
    fields = [normalize(field) for field in question.split("|")]

    if (
        len(fields) != 3
        or fields[0] != "find"
        or not fields[1]
        or fields[2] not in RELATIONS
    ):
        return "UNSUPPORTED", []

    subject, predicate = fields[1], fields[2]
    objects = index.get((subject, predicate), set())

    matches = [
        (subject, predicate, obj)
        for obj in sorted(objects)
    ]

    return ("OK" if matches else "NOT_FOUND"), matches


def readable_fact(fact, labels):
    subject, predicate, obj = fact
    return f"{labels[subject]} | {predicate} | {labels[obj]}"


def evaluate_sample(index):
    tests = [
        ("FIND | Python | is_a",
         "OK", {"programming language"}),
        ("FIND | Python | supports",
         "OK", {"oop"}),
        ("FIND | Python | enables",
         "OK", {"automation"}),
        ("FIND | System A | uses",
         "OK", {"python"}),
        ("FIND | Python | developed_by",
         "OK", {"guido van rossum"}),
        ("FIND | ML | subset_of",
         "OK", {"ai"}),
        ("FIND | DL | subset_of",
         "OK", {"ai", "ml"}),
        ("FIND | CNN | subset_of",
         "OK", {"ai", "dl", "ml"}),
        ("FIND | Java | uses",
         "NOT_FOUND", set()),
        ("FIND | Python | subset_of",
         "NOT_FOUND", set()),
        ("What is Python?",
         "UNSUPPORTED", set()),
        ("  find | python  | IS_A  ",
         "OK", {"programming language"}),
    ]

    results = []
    correct = 0

    for number, (question, expected_status, expected_values) in enumerate(
        tests, 1
    ):
        status, matches = ask(question, index)
        actual_values = {fact[2] for fact in matches}

        passed = (
            status == expected_status
            and actual_values == expected_values
        )
        correct += int(passed)

        results.append({
            "Test": f"T{number}",
            "Question": question,
            "Expected status": expected_status,
            "Actual status": status,
            "Expected values": "; ".join(sorted(expected_values)) or "-",
            "Actual values": "; ".join(sorted(actual_values)) or "-",
            "Result": "PASS" if passed else "FAIL",
        })

    return correct, results


# ---------------- WEBSITE INTERFACE ----------------

st.title("📄 AI-Based Document Question Answering System")
st.caption(
    "Ashitesh Gowda R | Register Number: 25ETAI410020 | "
    "AID201A — Artificial Intelligence"
)

st.write(
    "A symbolic AI demonstration using a knowledge base, "
    "exact fact lookup and rule-based inference."
)

with st.sidebar:
    st.header("About the project")
    st.write(
        "This application uses structured records and formal queries. "
        "It does not use NLP, language models, embeddings or RAG."
    )

    st.subheader("Reference websites")
    st.markdown(
        "- [Artificial Intelligence: A Modern Approach]"
        "(https://aima.cs.berkeley.edu/)\n"
        "- [Python Documentation]"
        "(https://docs.python.org/3/)\n"
        "- [pypdf Documentation]"
        "(https://pypdf.readthedocs.io/)\n"
        "- [python-docx Documentation]"
        "(https://python-docx.readthedocs.io/)"
    )

    st.warning(
        "Use demonstration documents only. Uploaded documents are "
        "processed on the hosting server; avoid confidential information."
    )

st.info(
    "Document format: subject | predicate | object. "
    "Question format: FIND | subject | predicate."
)

mode = st.radio(
    "Choose a document source",
    ["Sample document", "Upload a document"],
    horizontal=True,
)

st.download_button(
    "Download sample document",
    data=SAMPLE,
    file_name="sample.txt",
    mime="text/plain",
)

if mode == "Sample document":
    records = [
        (f"sample.txt: line {number}", line)
        for number, line in enumerate(SAMPLE.splitlines(), 1)
    ]

    with st.expander("View the sample document"):
        st.code(SAMPLE, language="text")
        st.caption(
            "ML = Machine Learning; DL = Deep Learning; "
            "CNN = Convolutional Neural Networks; "
            "AI = Artificial Intelligence; "
            "OOP = Object-Oriented Programming."
        )

else:
    uploaded = st.file_uploader(
        "Upload a structured TXT, text-based PDF or DOCX document",
        type=["txt", "pdf", "docx"],
    )

    st.caption(
        "Use one complete record per TXT/PDF line or DOCX paragraph. "
        "Ordinary prose, scanned PDFs and DOCX tables are not supported."
    )

    if uploaded is None:
        st.stop()

    try:
        records = read_uploaded_document(uploaded)
    except Exception as error:
        st.error(f"Document could not be read: {error}")
        st.stop()

try:
    facts, index, sources, parents, labels, direct_count = (
        build_knowledge_base(records)
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

st.success("Knowledge base constructed successfully.")

col1, col2, col3 = st.columns(3)
col1.metric("Direct facts", direct_count)
col2.metric("Inferred facts", len(facts) - direct_count)
col3.metric("Total facts", len(facts))

with st.expander("View the knowledge base"):
    st.dataframe([
        {
            "Subject": labels[subject],
            "Predicate": predicate,
            "Object": labels[obj],
            "Type": (
                "Inferred"
                if (subject, predicate, obj) in parents
                else "Direct"
            ),
        }
        for subject, predicate, obj in sorted(facts)
    ])

st.subheader("Ask a question")
st.caption(
    "Supported predicates: is_a, supports, enables, uses, "
    "developed_by, subset_of."
)

with st.form("question_form"):
    question = st.text_input(
        "Enter a formal question",
        value="FIND | DL | subset_of",
    )
    submitted = st.form_submit_button("Get answer")

if submitted:
    status, matches = ask(question, index)

    if status == "UNSUPPORTED":
        st.error(
            "UNSUPPORTED: use FIND | subject | predicate "
            "with one of the supported predicates."
        )

    elif status == "NOT_FOUND":
        st.warning(
            "NOT_FOUND: no supporting fact exists in this knowledge base. "
            "This does not prove that the relationship is false."
        )

    else:
        answer = "; ".join(labels[fact[2]] for fact in matches)
        st.success(f"Answer: {answer}")

        for fact in matches:
            with st.expander(
                "Evidence: " + readable_fact(fact, labels)
            ):
                if fact in parents:
                    left, right = parents[fact]
                    st.write("Derived using subset transitivity.")
                    st.code(
                        readable_fact(left, labels)
                        + "\nAND "
                        + readable_fact(right, labels)
                        + "\nTHEREFORE "
                        + readable_fact(fact, labels),
                        language="text",
                    )
                else:
                    st.write("Directly stated in the input document.")

                st.write("Supporting document locations:")
                for source in sorted(sources[fact]):
                    st.write("- " + source)

if mode == "Sample document":
    st.subheader("Evaluation")

    if st.button("Run the 12 sample tests"):
        correct, results = evaluate_sample(index)
        st.metric("Measured test accuracy", f"{correct / 12 * 100:.1f}%")
        st.write(f"Correct results: {correct}/12")
        st.dataframe(results)
        st.caption(
            "These results apply only to the supplied sample test set. "
            "They do not establish performance on arbitrary documents."
        )
