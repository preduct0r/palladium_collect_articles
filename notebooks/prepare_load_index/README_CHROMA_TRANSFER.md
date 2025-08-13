# Chroma Vector Index Transfer Guide

This guide explains how to create, transfer, and use a Chroma vector index between machines.

## Overview

The modified `chroma_index.py` script now creates a persistent Chroma vector database that can be transferred between machines. The index is stored on disk and can be packaged for easy transfer.

## Features

- ✅ **Persistent storage**: Index is saved to disk automatically
- ✅ **Incremental loading**: Reuses existing index if available
- ✅ **Transfer packaging**: Creates compressed archives for transfer
- ✅ **Metadata tracking**: Saves index information and creation details
- ✅ **Cross-platform**: Works on different machines with same embedding model

## Files

- `chroma_index.py` - Main script for creating/loading the index
- `load_chroma_index.py` - Utility for loading index on another machine
- `README_CHROMA_TRANSFER.md` - This guide

## Creating the Index

### 1. Run the main script

```bash
cd notebooks
python chroma_index.py
```

The script will:
- Create a persistent Chroma database in `./chroma_db/`
- Load and index your documents
- Save metadata about the index
- Offer to package the index for transfer

### 2. Package for transfer (optional)

When prompted, choose to package the index:
```
🤔 Хотите упаковать индекс для передачи на другую машину? (y/n): y
📁 Введите путь для архива (по умолчанию: chroma_index.tar.gz): my_index.tar.gz
```

## Transferring the Index

### Option 1: Transfer the archive file

If you created an archive:
```bash
# Copy the archive to the target machine
scp chroma_index.tar.gz user@target-machine:/path/to/destination/
```

### Option 2: Transfer the directory directly

```bash
# Copy the entire chroma_db directory
rsync -av chroma_db/ user@target-machine:/path/to/destination/chroma_db/
```

## Using the Index on Another Machine

### Prerequisites

Ensure the target machine has:
- Same Python environment with required packages
- Same embedding model server running (if using local embeddings)
- Access to the embedding API endpoint

### Method 1: Using the loader utility

```bash
# From archive
python load_chroma_index.py --from-archive chroma_index.tar.gz --interactive

# From directory
python load_chroma_index.py --index-path ./chroma_db --interactive

# Single query
python load_chroma_index.py --index-path ./chroma_db --query "your search query"
```

### Method 2: Using the main script

```bash
# Place the chroma_db directory in the correct location, then run:
python chroma_index.py
```

The script will detect the existing index and load it instead of creating a new one.

### Method 3: Programmatic usage

```python
from chroma_index import create_or_load_vectorstore, test_retriever

# Load existing vectorstore
vectorstore = create_or_load_vectorstore(persist_directory="./chroma_db")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Search
results = retriever.get_relevant_documents("your query")
```

## Configuration

### Embedding Model Configuration

Make sure the embedding model configuration matches between machines:

```python
# In both chroma_index.py and load_chroma_index.py
embedder = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B", 
    base_url="http://localhost:8090/v1",  # Update this for your setup
    api_key="EMPTY"
)
```

### Directory Paths

Update the `PERSIST_DIRECTORY` constant if needed:

```python
# In chroma_index.py
PERSIST_DIRECTORY = "/path/to/your/chroma_db"
```

## Command Line Examples

### Interactive Mode
```bash
python load_chroma_index.py --index-path ./chroma_db --interactive
```

### Single Query
```bash
python load_chroma_index.py --index-path ./chroma_db --query "palladium processing" --results 10
```

### Load from Archive
```bash
python load_chroma_index.py --from-archive chroma_index.tar.gz --target-dir ./my_chroma_db
```

### Custom Embedding Configuration
```bash
python load_chroma_index.py \
  --index-path ./chroma_db \
  --embedding-model "text-embedding-ada-002" \
  --base-url "https://api.openai.com/v1" \
  --api-key "your-api-key" \
  --interactive
```

## Troubleshooting

### Common Issues

1. **Embedding model mismatch**: Ensure the same embedding model is used on both machines
2. **Directory permissions**: Make sure the target directory is writable
3. **Missing dependencies**: Install all required packages on the target machine
4. **API endpoint not accessible**: Update the `base_url` for your embedding service

### Debugging

Check index information:
```python
from chroma_index import load_index_info
info = load_index_info("./chroma_db")
print(json.dumps(info, indent=2))
```

### Performance Notes

- Archive size depends on number of documents and embedding dimensions
- Transfer time depends on network speed and archive size
- Loading time depends on index size and disk speed

## Security Considerations

- Index files contain embedded vectors and metadata
- Ensure secure transfer methods for sensitive data
- Consider encryption for archive files if needed

## Example Workflow

```bash
# Machine 1: Create index
python chroma_index.py
# Choose 'y' to package for transfer

# Transfer
scp chroma_index.tar.gz user@machine2:/home/user/

# Machine 2: Load and use
python load_chroma_index.py --from-archive chroma_index.tar.gz --interactive
``` 