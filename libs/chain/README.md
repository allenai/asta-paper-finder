# ⛓️ ai2i.chain

A library for building and managing LLM (Large Language Model) chains in Python applications.

## Overview

`ai2i.chain` provides a structured and type-safe way to interact with LLMs, with features like:

- Type-safe prompt construction and response parsing
- Composable chain-based approach to LLM interactions
- Support for both prompt-based and chat-based LLM calls
- Integration with LangChain for robust LLM interactions
- Structured response handling with metadata

## Basic Usage

### Defining a Chat-based LLM Call

```python
from typing import TypedDict
from pydantic import BaseModel
from ai2i.chain import (
    define_chat_llm_call,
    system_message,
    user_message,
    ChatMessage
)

# Define input parameters
class SummarizeParams(TypedDict):
    text: str
    max_length: int

# Define expected output structure
class Summary(BaseModel):
    summary: str
    key_points: list[str]

# Define the LLM call
summarize = define_chat_llm_call(
    [
        system_message("You are a helpful assistant that summarizes text."),
        user_message("Please summarize the following text in {max_length} words or less:\n\n{text}")
    ],
    input_type=SummarizeParams,
    output_type=Summary
)

# Use the LLM call
result = await summarize.invoke({
    "text": "Long text to summarize...",
    "max_length": 100
})

print(result.summary)
print(result.key_points)
```

### Defining a Prompt-based LLM Call

```python
from typing import TypedDict
from pydantic import BaseModel
from ai2i.chain import define_prompt_llm_call

# Define input parameters
class ClassifyParams(TypedDict):
    text: str
    categories: list[str]

# Define expected output structure
class Classification(BaseModel):
    category: str
    confidence: float
    reasoning: str

# Define the LLM call
classify = define_prompt_llm_call(
    """
    Classify the following text into one of these categories: {categories}

    Text: {text}

    Provide your classification along with confidence level and reasoning.
    """,
    input_type=ClassifyParams,
    output_type=Classification
)

# Use the LLM call
result = await classify.invoke({
    "text": "The stock market showed significant gains today.",
    "categories": ["Politics", "Business", "Technology", "Sports"]
})

print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

## Advanced Features

### Including Response Metadata

You can include response metadata (like token usage) in the LLM call results:

```python
from ai2i.chain import define_chat_llm_call, ResponseMetadata

summarize_with_metadata = define_chat_llm_call(
    [...],  # messages
    input_type=SummarizeParams,
    output_type=Summary,
    include_response_metadata=True
)

# Result will be a tuple of (Summary, ResponseMetadata)
result, metadata = await summarize_with_metadata.invoke({...})

# Access metadata
print(f"Tokens used: {metadata.get('token_usage', {}).get('total_tokens')}")
```

### Custom Template Formats

The library supports different template formats:

```python
from ai2i.chain import define_prompt_llm_call

# Using Mustache templates
classify_mustache = define_prompt_llm_call(
    """
    Classify the following text into one of these categories: {{categories}}

    Text: {{text}}
    """,
    input_type=ClassifyParams,
    output_type=Classification,
    format="mustache"
)
```

## LLM Endpoints

The library provides a way to define and use different LLM endpoints:

```python
from ai2i.chain import define_llm_endpoint, LLMModel, ModelName, Timeouts

# Define an endpoint
openai_endpoint = define_llm_endpoint(
    model=LLMModel(name=ModelName.GPT_4),
    timeouts=Timeouts(
        connect_timeout=10.0,
        read_timeout=60.0
    )
)

# Use the endpoint in your application context
with application_context(llm_endpoint=openai_endpoint):
    result = await summarize.invoke({...})
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
