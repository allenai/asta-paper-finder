# Paper Finder

This repo contains the code for the standalone Paper Finder agent (namely PaperFinder).

This code is not actively maintained, and reflects a snapshot in time -- of the online/live PaperFinder agent -- which is easy to run locally. This is also the code that is used for evaluating PaperFinder and that can be used for reproducing the reported results.

PaperFinder is our paper-seeking agent, which is intended to assist in locating sets of papers according to content-based and metadata criteria.

### High Level Overview

PaperFinder is implemented as a pipeline of manual-coded components which involve LLM decisions in several key-points, as well as LLM-based relevance judgments of retrieved abstracts and snippets. At a high-level, a query is analyzed and transformed into a structured object which is then fed to an execution planner that routes the analyzed query to one of several workflows, each covering a particular paper-seeking intent. Each workflow may involve multiple steps, and returns a relevance-judged set of papers, which is then ranked while weighting content relevance together with other criteria which may appear in the query (e.g., "early works on", "influential" etc).

### How does this version differ from the live version?

This code departs from the PaperFinder agent which is [deployed online](https://paperfinder.allen.ai/). The online agent has several abilities that are not supported by the agent in this repo (and which are not relevant for the evaluation), such as the ability to handle multi-turn interaction with a user, to send user-friendly progress updates to a host environment, to maintain conversations over long periods of time and to show a graphical widget with the results. It also consults datasets and indices which are legal for us to use in the product but not in the public-facing API. The online agent also differs from this code in some of its configuration options, as evaluation setup differs from product setups (for example, in a product it is OK or even preferable to also show in the results papers that are ranked right after the relevant ones, to ask a user for clarifications, or to refuse some queries which seem out of scope, all of which are not advised in an evaluation setup). Finally, the online agent keeps updating regularly and is tightly integrated in the production environment. For this release we wanted a stable, consistent version, which focuses on the core capability of paper-finding given a single user query. We plan to release larger chunks of PaperFinder agent, in particular the multi-turn abilities, as they become more mature and stable, and as we have proper benchmarks for them.

This code was created by cloning the internal PaperFinder repo and brutally removing various environment, UI and conversation management related code, some internal search APIs and so on, while keeping the single-turn paper-search functionality working and effective.

## How to run

### secrets file

The agent requires multiple keys, which should be defined in a `.env.secret` file under `agents/mabool/api/conf`.
The needed keys are: `OPENAI_API_KEY`, `S2_API_KEY`, `COHERE_API_KEY`, `GOOGLE_API_KEY`.

### Environment

We are using [uv](https://docs.astral.sh/uv/#highlights) as our project manager, so to prepare the environement:

```bash
make sync-dev
```

### Running
To run the agent, we launch a FastAPI server. Once the server is running, we can interact with it using cURL from the command line or through the Swagger web interface.

```bash
cd agents/mabool/api
make start-dev
```

### API

The `/api/2/rounds` POST API has the following arguments:

```Json
{
  "paper_description": "string",
  "operation_mode": "infer",
  "inserted_before": "string",
  "read_results_from_cache": false
}
```
- paper_description (REQUIRED): The natural language paper-search description.
- operation_mode (default="infer"): should be one of "infer", "fast" or "diligent". Currently "infer" and "fast" behave the same way, and initiate a fast search (~30 seconds) whereas the "diligent" mode does a more exhaustive fetching (~3 minutes).
- inserted_before (default=None): an upper-bound date in the format of YYYY-MM-DD to limit the results
- read_results_from_cache (default=False): We cache the results on disk, and allow returning the results if they are found in that file-based cache.
