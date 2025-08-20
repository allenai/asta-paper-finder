# ASTA Paper Finder

This repo contains the code for the standalone ASTA Paper Finder agent (namely PaperFinder).

This code is not actively maintained, and reflects a snapshot in time -- of the online/live PaperFinder agent -- which is easy to run locally. This is also the code that is used for evaluating PaperFinder in AstaBench and that should be used for reproducibility.

### Description

PaperFinder is our paper-seeking agent, which is intended to assist in locating sets of papers according to content-based and metadata criteria. It is implemented as a pipeline of manual-coded components which involve LLM decisions in several key-points, as well as LLM-based relevance judgments of retrieved abstracts and snippets. At a high-level, a query is analyzed and transformed into a structured object which is then fed to an execution planner that routes the analyzed query to one of several workflows, each covering a particular paper-seeking intent. Each workflow may involve multiple steps, and returns a relevance-judged set of papers, which is then ranked while weighting content relevance together with other criteria which may appear in the query (e.g., "early works on", "influential" etc). It is described in more details in AstaBench paper (link/cite TBD).

### Differences from live PaperFinder

This code departs from the PaperFinder agent which is [deployed online](https://asta.allen.ai/) in the Asta project. The online agent has several abilities that are not supported by the agent in this repo (and which are not relevant for the AstaBench evaluation), such as the ability to handle multi-turn interaction with a user, to send user-friendly progress updates to a host environment, to maintain conversations over long periods of time and to show a graphical widget with the results. It also consults datasets and indices which are legal for us to use in the product but not in the public-facing API provided for AstaBench. The online agent also differs from this code in some of its configuration options, as evaluation setup differs from product setups (for example, in a product it is OK or even preferable to also show in the results papers that are ranked right after the relevant ones, to ask a user for clarifications, or to refuse some queries which seem out of scope, all of which are not advised in an evaluation setup). Finally, the online agent keeps updating regularly and is tightly integrated in the production environment. For this release we wanted a stable, consistent version, which focuses on the core capability of paper-finding given a single user query. We intend to release larger chunks of PaperFinder agent, in particular the multi-turn abilities, as they become more mature and stable, and as we have proper benchmarks for them.

This code was created by cloning the internal PaperFinder repo and brutally removing the less-relevant sections mentioned above. 


## How to run

To run the agent, we launch a FastAPI server, either by executing the main Python application directly or by deploying it with Gunicorn. Once the server is running, we can interact with it using cURL from the command line or through the Swagger web interface.
### Run the python script directly 
```bash
python agents/mabool/api/mabool/api/app.py
```

### Run the bash script
```bash
bash agents/mabool/api/mabool/api/dev.sh
```

### secrets file

The agent requires multiple keys, which should be defined in a `.env.secret` file under `agents/mabool/api/conf`.
The needed keys are: `OPENAI_API_KEY`, `S2_API_KEY`, `COHERE_API_KEY`, `GOOGLE_API_KEY`.
