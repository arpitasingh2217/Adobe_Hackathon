# PDF Heading Extractor

This solution extracts structured outlines from PDF documents, identifying the document title and heading hierarchy (H1, H2, H3) with their respective page numbers.

## Requirements

- Python 3.9+
- PyMuPDF 1.26.3+

## Docker Usage

### Building the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-extractor:v1 .
```

### Running the Container

```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none pdf-extractor:v1
```
