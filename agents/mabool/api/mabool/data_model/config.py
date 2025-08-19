from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai2i.config.config_models import ConfigValuePlaceholder


@dataclass(frozen=True)
class LlmSuggestionAgent:
    llm_model_name: ConfigValuePlaceholder[Literal["google:gemini2flash-default"]] = ConfigValuePlaceholder(
        ["llm_suggestion_agent", "llm_model_name"]
    )
    openai_concurrency: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["llm_suggestion_agent", "openai_concurrency"]
    )
    fallback_n_suggestions: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["llm_suggestion_agent", "fallback_n_suggestions"]
    )


@dataclass(frozen=True)
class SearchByAuthorAgent:
    llm_model_name: ConfigValuePlaceholder[Literal["google:gemini2flash-default"]] = ConfigValuePlaceholder(
        ["search_by_author_agent", "llm_model_name"]
    )
    relevance_judgements_quota: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["search_by_author_agent", "relevance_judgements_quota"]
    )
    limit_for_specific: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["search_by_author_agent", "limit_for_specific"]
    )
    limit_for_broad: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["search_by_author_agent", "limit_for_broad"])
    disambiguate_authors: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(
        ["search_by_author_agent", "disambiguate_authors"]
    )
    consider_profiles_per_author: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["search_by_author_agent", "consider_profiles_per_author"]
    )


@dataclass(frozen=True)
class SpecificPaperByNameAgent:
    should_sort: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["specific_paper_by_name_agent", "should_sort"])
    filter_threshold: ConfigValuePlaceholder[float] = ConfigValuePlaceholder(
        ["specific_paper_by_name_agent", "filter_threshold"]
    )


@dataclass(frozen=True)
class SpecificPaperByTitleAgent:
    llm_model_name: ConfigValuePlaceholder[Literal["google:gemini2flash-default"]] = ConfigValuePlaceholder(
        ["specific_paper_by_title_agent", "llm_model_name"]
    )
    should_sort: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["specific_paper_by_title_agent", "should_sort"])


@dataclass(frozen=True)
class BroadSearchByKeywordAgent:
    results_limit: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["broad_search_by_keyword_agent", "results_limit"]
    )
    extra_results_factor: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["broad_search_by_keyword_agent", "extra_results_factor"]
    )
    formulation_model_name: ConfigValuePlaceholder[Literal["google:gemini2flash-default"]] = ConfigValuePlaceholder(
        ["broad_search_by_keyword_agent", "formulation_model_name"]
    )


@dataclass(frozen=True)
class MetadataPlannerAgent:
    llm_model_name: ConfigValuePlaceholder[Literal["openai:gpt4o-default"]] = ConfigValuePlaceholder(
        ["metadata_planner_agent", "llm_model_name"]
    )
    ops_max_concurrency: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["metadata_planner_agent", "ops_max_concurrency"]
    )


@dataclass(frozen=True)
class QueryAnalyzerAgent:
    llm_abstraction_model_name: ConfigValuePlaceholder[Literal["google:gemini2flash-default"]] = ConfigValuePlaceholder(
        ["query_analyzer_agent", "llm_abstraction_model_name"]
    )
    force_broad: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["query_analyzer_agent", "force_broad"])


@dataclass(frozen=True)
class RelevanceJudgement:
    relevance_model_name: ConfigValuePlaceholder[Literal["google:gemini2flash-default"]] = ConfigValuePlaceholder(
        ["relevance_judgement", "relevance_model_name"]
    )
    openai_concurrency: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["relevance_judgement", "openai_concurrency"]
    )
    quota: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["relevance_judgement", "quota"])
    highly_relevant_cap: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["relevance_judgement", "highly_relevant_cap"]
    )
    window_size: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["relevance_judgement", "window_size"])
    initial_batch_size: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["relevance_judgement", "initial_batch_size"]
    )
    uniform_preload_size: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["relevance_judgement", "uniform_preload_size"]
    )
    decay_factor: ConfigValuePlaceholder[float] = ConfigValuePlaceholder(["relevance_judgement", "decay_factor"])
    batch_growth_factor: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["relevance_judgement", "batch_growth_factor"]
    )
    plot: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["relevance_judgement", "plot"])
    optimal_solution: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["relevance_judgement", "optimal_solution"])
    keep_irrelevant_docs: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(
        ["relevance_judgement", "keep_irrelevant_docs"]
    )


@dataclass(frozen=True)
class BroadSearchAgent:
    max_iterations: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["broad_search_agent", "max_iterations"])
    llm_n_suggestions: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["broad_search_agent", "llm_n_suggestions"])


@dataclass(frozen=True)
class DenseAgent:
    formulation_model_name: ConfigValuePlaceholder[Literal["google:gemini2flash-default"]] = ConfigValuePlaceholder(
        ["dense_agent", "formulation_model_name"]
    )
    initial_queries_to_formulate: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["dense_agent", "initial_queries_to_formulate"]
    )
    initial_top_k_per_query: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["dense_agent", "initial_top_k_per_query"]
    )
    reformulate_prompt_num_queries: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["dense_agent", "reformulate_prompt_num_queries"]
    )
    reformulate_prompt_example_docs: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["dense_agent", "reformulate_prompt_example_docs"]
    )
    reformulate_top_k: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["dense_agent", "reformulate_top_k"])
    dense_agent_max_iterations: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["dense_agent", "dense_agent_max_iterations"]
    )


@dataclass(frozen=True)
class Vespa:
    concurrency: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["vespa", "concurrency"])
    timeout: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["vespa", "timeout"])


@dataclass(frozen=True)
class LlmAbstraction:
    gpt4o_default_model: ConfigValuePlaceholder[Literal["gpt-4o-2024-11-20"]] = ConfigValuePlaceholder(
        ["llm_abstraction", "gpt4o_default_model"]
    )
    gpt4turbo_default_model: ConfigValuePlaceholder[Literal["gpt-4-turbo"]] = ConfigValuePlaceholder(
        ["llm_abstraction", "gpt4turbo_default_model"]
    )
    gpt4_default_model: ConfigValuePlaceholder[Literal["gpt-4-0125-preview"]] = ConfigValuePlaceholder(
        ["llm_abstraction", "gpt4_default_model"]
    )
    claude37sonnet_default_model: ConfigValuePlaceholder[Literal["claude-3-7-sonnet-20250219"]] = (
        ConfigValuePlaceholder(["llm_abstraction", "claude37sonnet_default_model"])
    )
    gemini2flash_default_model: ConfigValuePlaceholder[Literal["gemini-2.0-flash"]] = ConfigValuePlaceholder(
        ["llm_abstraction", "gemini2flash_default_model"]
    )
    temparature: ConfigValuePlaceholder[float] = ConfigValuePlaceholder(["llm_abstraction", "temparature"])
    n: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["llm_abstraction", "n"])
    batch_max_concurrency: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["llm_abstraction", "batch_max_concurrency"]
    )


@dataclass(frozen=True)
class Di:
    round_scope_timeout: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["di", "round_scope_timeout"])


@dataclass(frozen=True)
class Cache:
    enabled: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["cache", "enabled"])
    ttl: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["cache", "ttl"])


@dataclass(frozen=True)
class Cohere:
    rerank_model_name: ConfigValuePlaceholder[Literal["rerank-english-v3.0"]] = ConfigValuePlaceholder(
        ["cohere", "rerank_model_name"]
    )


@dataclass(frozen=True)
class S2Api:
    concurrency: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["s2_api", "concurrency"])
    timeout: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["s2_api", "timeout"])
    total_papers_limit: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["s2_api", "total_papers_limit"])
    total_citations_limit: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["s2_api", "total_citations_limit"])
    retry_attempts: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["s2_api", "retry_attempts"])
    max_seconds_to_wait: ConfigValuePlaceholder[float] = ConfigValuePlaceholder(["s2_api", "max_seconds_to_wait"])


@dataclass(frozen=True)
class SnowballAgent:
    forward_top_k: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["snowball_agent", "forward_top_k"])
    backward_top_k: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["snowball_agent", "backward_top_k"])
    snippet_top_k: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["snowball_agent", "snippet_top_k"])


@dataclass(frozen=True)
class FastBroadSearchAgent:
    dense_top_k: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["fast_broad_search_agent", "dense_top_k"])
    snowball_snippets_top_k: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["fast_broad_search_agent", "snowball_snippets_top_k"]
    )
    s2_relevance_search_top_k: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["fast_broad_search_agent", "s2_relevance_search_top_k"]
    )
    relevance_judgement_quota: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(
        ["fast_broad_search_agent", "relevance_judgement_quota"]
    )


@dataclass(frozen=True)
class AppConfigSchema:
    fast_broad_search_agent: FastBroadSearchAgent = FastBroadSearchAgent()
    snowball_agent: SnowballAgent = SnowballAgent()
    s2_api: S2Api = S2Api()
    cohere: Cohere = Cohere()
    cache: Cache = Cache()
    di: Di = Di()
    llm_abstraction: LlmAbstraction = LlmAbstraction()
    vespa: Vespa = Vespa()
    dense_agent: DenseAgent = DenseAgent()
    broad_search_agent: BroadSearchAgent = BroadSearchAgent()
    relevance_judgement: RelevanceJudgement = RelevanceJudgement()
    query_analyzer_agent: QueryAnalyzerAgent = QueryAnalyzerAgent()
    metadata_planner_agent: MetadataPlannerAgent = MetadataPlannerAgent()
    broad_search_by_keyword_agent: BroadSearchByKeywordAgent = BroadSearchByKeywordAgent()
    specific_paper_by_title_agent: SpecificPaperByTitleAgent = SpecificPaperByTitleAgent()
    specific_paper_by_name_agent: SpecificPaperByNameAgent = SpecificPaperByNameAgent()
    search_by_author_agent: SearchByAuthorAgent = SearchByAuthorAgent()
    llm_suggestion_agent: LlmSuggestionAgent = LlmSuggestionAgent()
    env: ConfigValuePlaceholder[Literal["dev"]] = ConfigValuePlaceholder(["env"])
    log_format: ConfigValuePlaceholder[Literal[""]] = ConfigValuePlaceholder(["log_format"])
    log_max_length: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["log_max_length"])
    response_text_top_k: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["response_text_top_k"])
    run_snowball_for_recent: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["run_snowball_for_recent"])
    run_snowball_for_central: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["run_snowball_for_central"])
    assume_recent_and_central_first: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(
        ["assume_recent_and_central_first"]
    )
    consider_original_order: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["consider_original_order"])
    should_sort: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["should_sort"])
    force_deterministic: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["force_deterministic"])
    enable_llm_cache: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["enable_llm_cache"])
    operative_timeout: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["operative_timeout"])
    show_api_swagger: ConfigValuePlaceholder[bool] = ConfigValuePlaceholder(["show_api_swagger"])
    time_to_wait_for_results: ConfigValuePlaceholder[int] = ConfigValuePlaceholder(["time_to_wait_for_results"])

    def __getattr__(self, name: str) -> ConfigValuePlaceholder[str]:
        return ConfigValuePlaceholder([name])


cfg_schema = AppConfigSchema()
