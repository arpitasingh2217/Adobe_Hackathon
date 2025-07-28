# Adobe_Hackathon
# PDF Outline Extractor

#### A Dockerized solution that extracts hierarchical document outlines (title and headings) from PDF files and outputs structured JSON.

## APPROACH
This solution processes PDF documents to:
1. Identify and extract the document title
2. Detect heading structures (H1, H2, H3 levels)
3. Record the page number where each heading appears
4. Output a clean JSON representation of the document outline

The extractor is designed to work with standard PDF documents up to 50 pages, with no internet connectivity required.

## MODELS AND LIBRARIES USED
- **PyPDF2**: For PDF text extraction and page navigation
- **Python 3.9**: As the base runtime environment
- **Docker**: For containerization and cross-platform compatibility

## SYSTEM REQUIREMENTS
- Docker installed
- AMD64 (x86_64) architecture
- 200MB available disk space

## How to Build and RuN
FROM --platform=linux/amd64 python:3.9-slim

##### 1.Set working directory
WORKDIR /app

##### 2.Install system dependencies first (cached layer)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

#### 3.Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#### 4.Copy source code
COPY extractor.py .

#### 5.Create directories
RUN mkdir -p /app/input /app/output

#### 6.Set the entry point
CMD ["python", "extractor.py"]

#### Run Docker Image
``bash
docker build --platform linux/amd64 -t pdf-extractor:v1 .

#### 7.INPUT
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-outline-extractor:latest

#### 8.OUTPUT
{
  "title": "Document Title",
  
  "outline": [
  
    { "level": "H1", "text": "Main Heading", "page": 1 },
    
    { "level": "H2", "text": "Subheading", "page": 2 },
    
    { "level": "H3", "text": "Nested Item", "page": 3 }
    
  ]
  
}


### TECHNICAL SPECIFICATIONS

Platform: AMD64 (x86_64) compatible

Architecture: CPU-only (no GPU requirements)

Network: Works completely offline

Model Size: < 200MB

Maximum PDF size: 50 pages
