
import os
from pathlib import Path

from aws.s3_storage import S3Storage


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_local_documents(folder):
    """Load text and Markdown documents from a local folder."""

    folder = Path(folder)
    documents = []

    if not folder.exists():
        raise FileNotFoundError(
            f"Local document folder not found: {folder}"
        )

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append({
                "source": str(path),
                "text": path.read_text(encoding="utf-8")
            })

    return documents


def load_s3_documents(prefix="documents/"):
    """Read supported text documents from an existing S3 bucket."""

    storage = S3Storage()
    documents = []

    keys = storage.list_objects(prefix=prefix)

    for key in keys:
        extension = Path(key).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        text = storage.read_text(key)

        documents.append({
            "source": f"s3://{storage.bucket_name}/{key}",
            "text": text
        })

    return documents


def load_documents(local_folder=None, use_s3=False):
    """Load documents from S3 or use local fallback."""

    if use_s3:
        return load_s3_documents()

    if local_folder is None:
        local_folder = (
            Path(__file__).resolve().parent / "documents"
        )

    return load_local_documents(local_folder)


if __name__ == "__main__":

    use_s3 = (
        os.getenv("USE_S3", "false").lower() == "true"
    )

    documents = load_documents(use_s3=use_s3)

    print("Documents loaded:", len(documents))

    for document in documents:
        print("Source:", document["source"])
        print("Characters:", len(document["text"]))
