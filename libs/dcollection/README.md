# 🗃️ ai2i.dcollection

A library for managing and manipulating collections of documents, particularly academic papers, with a focus on flexible data access, transformation, and retrieval.

## Overview

`ai2i.dcollection` provides a structured and type-safe way to work with document collections, with features like:

- Type-safe document field access and manipulation
- Dynamic loading of document fields on demand
- Flexible document collection operations (filtering, mapping, grouping, etc.)
- Integration with various data sources, particularly Semantic Scholar
- Support for computed fields and document transformations
- Caching mechanisms for efficient data access
- Conversion to pandas DataFrames for analysis

## Basic Usage

### Creating a Document Collection

```python
from ai2i.dcollection import DocumentCollectionFactory, PaperFinderDocumentCollection

# Create a factory with an S2 API key
factory = DocumentCollectionFactory(s2_api_key="your-s2-api-key-here")

# Create a collection from Semantic Scholar paper IDs
corpus_ids = ["1234567890", "0987654321"]
collection = factory.from_ids(corpus_ids)

# Load specific fields for all documents in the collection
collection = collection.with_fields(["title", "abstract", "authors"])

# Print the titles of all papers in the collection
for doc in collection:
    print(doc["title"])
```

### Searching for Papers

```python
from ai2i.dcollection import DocumentCollectionFactory

factory = DocumentCollectionFactory()

# Search for papers by title
title_collection = factory.from_s2_by_title("machine learning")

# Search for papers by author
author_collection = factory.from_s2_by_author(["John Smith"], limit=10)

# Search for papers using Semantic Scholar search
search_collection = factory.from_s2_search(
    query="natural language processing",
    limit=20,
    time_range={"start_year": 2020, "end_year": 2023},
    venues=["ACL", "EMNLP"]
)
```

### Manipulating Collections

```python
from ai2i.dcollection import PaperFinderDocumentCollection

# Filter a collection
recent_papers = collection.filter(lambda doc: doc["year"] >= 2020)

# Sort a collection
sorted_papers = collection.sorted([
    {"field": "year", "order": "desc"},
    {"field": "citation_count", "order": "desc"}
])

# Take the first N papers
top_papers = collection.take(10)

# Group papers by venue
papers_by_venue = collection.group_by(lambda doc: doc["venue"])

# Convert to a pandas DataFrame for analysis
df = collection.to_dataframe(["title", "abstract", "year", "citation_count"])
```

## Advanced Features

### Working with Computed Fields

```python
from ai2i.dcollection import ComputedField, PaperFinderDocumentCollection
from typing import Any

# Define a computed field
class TitleLength(ComputedField):
    def computation(self, doc: Any) -> int:
        return len(doc["title"])

# Add the computed field to a collection
collection_with_title_length = collection.update_computed_fields([TitleLength(name="title_length")])

# Use the computed field
for doc in collection_with_title_length:
    print(f"Title: {doc['title']}")
    print(f"Title length: {doc['title_length']}")
```

#### Alternative Method: Defining Computed Fields Directly with with_fields

You can also define computed fields directly when loading fields with `with_fields`:

```python
from ai2i.dcollection import ComputedField, PaperFinderDocumentCollection
from typing import Any

# Define a computed field
class CitationRatio(ComputedField):
    def computation(self, doc: Any) -> float:
        if doc["reference_count"] == 0:
            return 0
        return doc["citation_count"] / doc["reference_count"]

# Load fields and add computed field in one step
collection_with_ratio = await collection.with_fields([
    "title", "citation_count", "reference_count", 
    CitationRatio(name="citation_ratio")
])

# Use the computed field
for doc in collection_with_ratio:
    print(f"Paper: {doc['title']}")
    print(f"Citation ratio: {doc['citation_ratio']:.2f}")
```

#### Types of Computed Fields

The library provides several types of computed fields for different use cases:

##### 1. ComputedField

A simple computed field that operates on a single document. Use this when:
- The computation doesn't require async operations
- The value can be derived from other fields in the same document
- The computation is independent for each document

```python
from ai2i.dcollection import ComputedField

class WordCount(ComputedField):
    def computation_func(self, doc):
        # Simple synchronous computation on a single document
        return len(doc["abstract"].split()) if doc["abstract"] else 0

# By default, ComputedField results are cached
word_count = WordCount(name="word_count", cache=True)
```

##### 2. BatchComputedField

For computations that need to process multiple documents at once, especially for async operations. Use this when:
- The computation involves I/O or external API calls
- Batching improves performance
- Each document's result is still independent

```python
from ai2i.dcollection import BatchComputedField
import aiohttp

class ExternalSentimentScore(BatchComputedField):
    async def computation_func(self, docs):
        # Process documents in a batch with async operations
        async with aiohttp.ClientSession() as session:
            results = []
            for doc in docs:
                async with session.post(
                    "https://api.sentiment-analysis.com/analyze",
                    json={"text": doc["abstract"]}
                ) as response:
                    data = await response.json()
                    results.append(data["score"])
            return results

sentiment_field = ExternalSentimentScore(name="sentiment", cache=True)
```

##### 3. AggTransformComputedField

For computations where a document's value depends on other documents in the collection. Use this when:
- The computation requires context from the entire collection
- Values need to be normalized or ranked relative to other documents

```python
from ai2i.dcollection import AggTransformComputedField

class RelativeImpactScore(AggTransformComputedField):
    async def computation_func(self, docs):
        # Compute a score that depends on all documents in the batch
        max_citations = max(doc["citation_count"] or 0 for doc in docs)
        if max_citations == 0:
            return [0.0] * len(docs)

        # Normalize citation counts relative to the maximum
        return [
            (doc["citation_count"] or 0) / max_citations
            for doc in docs
        ]

# AggTransformComputedField defaults to no caching since values depend on the collection
relative_impact = RelativeImpactScore(name="relative_impact", cache=False)
```

##### 4. AssignedField

For values that are computed externally and can't be recomputed from other fields. Use this when:
- Values come from an external source or computation
- The computation can't be reproduced for new documents
- Values need to be assigned in a specific order

```python
from ai2i.dcollection import AssignedField

# Create a field with pre-computed values
relevance_scores = [0.95, 0.87, 0.76, 0.65, 0.54]
relevance_field = AssignedField(
    name="relevance_score",
    assigned_values=relevance_scores
)

# Assign the values to documents
relevance_field.values_to_docs(collection.documents)
```

### Understanding Document Data Management

In dcollection, there are three key components for managing document data:

- **Fetchers**: Functions for fetching new documents from external sources
- **Loaders**: Functions for filling additional/missing field data for existing documents in a collection
- **Fusers**: Functions for merging the same documents from different collections

#### Fetchers and External Sources

Fetchers retrieve new documents from external sources. Currently supported sources include:

- **Semantic Scholar**: Academic paper metadata and full-text access
- **Dense Retrieval**: Vector-based document retrieval, via Bifroest (internal ai2i) and Vespa (public)
- **SPIKE**: Syntactic search engine

```python
# Example of fetching documents from Semantic Scholar
papers = factory.from_s2_search("natural language processing", limit=10)

# Example of fetching documents using dense retrieval
papers = factory.from_dense_retrieval(
    queries=["transformer architecture"], 
    dataset=DenseDataset(name="s2-corpus", provider="bifroest"),
    top_k=20,
    search_iteration=1
)
```

#### Loaders and Field Definitions

Loaders fill in missing field data for documents that already exist in a collection. Here's a subset of field definitions from the library:

```python
# Field definitions from document.py
class PaperFinderDocument(Document):
    corpus_id: CorpusId = Field()
    ...
    title: str | None = dynamic_field(default=None, loaders=[from_s2])
    authors: list[Author] | None = dynamic_field(default=None, loaders=[from_s2])
    abstract: str | None = dynamic_field(default=None, loaders=[from_s2])
    citation_count: int | None = dynamic_field(default=None, loaders=[from_s2])
    snippets: list[Snippet] | None = dynamic_field(default=None, fuse=fuse_snippet)
    markdown: str | None = dynamic_field(
        default=None,
        loaders=[load_markdown],
        required_fields=[
            "title", "authors", "year", "abstract", 
            "snippets", "citation_contexts",
        ],
    )
    ...
```

#### Fusers for Merging Documents

Fusers combine data from the same document across different collections:

```python
from ai2i.dcollection import DynamicField, DocumentFieldLoader, Fuser

# Example of how fusers work
def fuse_snippet(fuse_to, fuse_from, field):
    """Merge snippets from two versions of the same document."""
    fuse_to_snippets = getattr(fuse_to, field) if fuse_to.is_loaded(field) else []
    fuse_from_snippets = getattr(fuse_from, field) if fuse_from.is_loaded(field) else []

    if fuse_to_snippets or fuse_from_snippets:
        # Merge and deduplicate snippets
        merged_snippets = _merge_snippets(fuse_to_snippets + fuse_from_snippets)
        setattr(fuse_to, field, merged_snippets)

# Fields can be defined with custom loaders and fusers
dynamic_field = DynamicField(
    loaders=[my_custom_loader],  # Functions that load field data
    fuse=fuse_snippet,           # Function that merges field data
    required_fields=["title"],   # Fields required for computation
    cache=True                   # Whether to cache the results
)
```

### Merging Collections

```python
from ai2i.dcollection import PaperFinderDocumentCollection

# Merge collections
merged_collection = collection1.merged(collection2, collection3)

# Alternative syntax
merged_collection = collection1 + collection2 + collection3

# Subtract collections
difference_collection = collection1 - collection2
```

### Sampling Documents

```python
from ai2i.dcollection import PaperFinderDocumentCollection, SampleMethod

# Random sampling
random_sample = collection.sample(10, method=SampleMethod.RANDOM)

# Stratified sampling
stratified_sample = collection.sample(10, method=SampleMethod.STRATIFIED)
```

## Development

### Setup Development Environment

```shell script
make sync-dev
```

### Testing

```shell script
make test
# -or-
make test-cov  # With coverage
```

### Code Quality

```shell script
make style  # Run format check, lint and type-check
make fix    # Automatically fix style issues
```
