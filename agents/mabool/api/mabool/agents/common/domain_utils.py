from __future__ import annotations

from typing import Mapping

from ai2i.common.utils.data_struct import SortedSet
from ai2i.dcollection import DenseDataset

from mabool.data_model.agent import DomainsIdentified


def get_system_domain_params(domains: DomainsIdentified) -> Mapping[str, str]:
    if "Computer Science" in [domains.main_field] + domains.key_secondary_fields:
        return {
            "domain_description": "linguistics, math, computer science, machine learning, or artificial intelligence (natural language processing or computer vision in particular)",
            "determiner_example": '"model" or "dataset" (for example, for "the BERT model" extract only "BERT")',
            "affiliation_example": 'for a github repo "huggingface/transformers" extract only "transformers"',
        }
    else:
        return {
            "domain_description": ", ".join([domains.main_field] + domains.key_secondary_fields),
            "determiner_example": '"region" (for example, for "the hippocampus region" extract only "hippocampus")',
            "affiliation_example": "for a medicine name Eliquis (Pfizer) extract only Eliquis",
        }


def get_dense_datasets_by_domains(domains: DomainsIdentified) -> list[DenseDataset]:
    return [DenseDataset("vespa", "open-nora", "pa1-v1")]


def get_fields_of_study_filter_from_domains(domains: DomainsIdentified) -> list[str]:
    # NOTE - for now we assume most of our usage would be from CS domain and thus always add it to the FoS filter
    full_list = list(SortedSet(["Computer Science"] + [domains.main_field] + domains.key_secondary_fields))
    # remove "Unknown" and empty strings which S2 API doesn't know
    return [f for f in full_list if f and f != "Unknown"]
