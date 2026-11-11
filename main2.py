import glob
import json
import math
import os
import re
from typing import Literal

import pydantic
from bs4 import BeautifulSoup
from opentelemetry import trace

from evals.base_evals import BaseEvaluator, EvalResult
from prompt_chains.base_chains import BaseChain, ChainResult, Document
from prompt_chains.long_html_splitter import HTMLSplitter
from prompt_chains.routing_chains_bool import (
    BoolRoute,
    BoolRouterChain,
    BoolRoutingCondition,
)
from utils.myOllama import ollamaChat
from utils.otel_setup import setup_otel_tracing
from utils.llm_models import ChatAdapter


class HTMLRelevance(BaseChain):
    def __init__(
        self,
        llm: ChatAdapter,
        evaluators: list[BaseEvaluator] | None = None,
        eval_max_retries: int = 0,
        invoke_max_retries: int = 0,
    ):
        super().__init__(
            evaluators=evaluators,
            eval_max_retries=eval_max_retries,
            invoke_max_retries=invoke_max_retries,
        )
        self.llm = llm

    def _invoke(self, docs: list[Document]) -> ChainResult:

        system_prompt = """
        Does the snippet of html provided contain information about the outstanding common shares of the company?

        Outstanding common shares refers to the total number of shares AFTER any IPO or corporate actions.

        Share offered in an IPO should NOT be counted as outstanding shares.
        """

        if len(docs) != 1:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=f"Expected exactly 1 document as input, but got {len(docs)}",
            )

        tags_fields = {"Relevance": (Literal["YES", "NO"], ...)}

        ResponseModel = pydantic.create_model("ResponseModel", **tags_fields)

        payload = {
            "think": False,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": docs[0].page_content},
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0},
        }

        categorisation_result = self.llm.chat_with_expected_type_result(
            chat_payload=payload, result_type="dict"
        )

        if not categorisation_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=categorisation_result.error_message,
            )

        if not categorisation_result.model_result.get("Relevance") in ["YES", "NO"]:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message="Output result is not one of ['YES', 'NO']",
            )

        if categorisation_result.model_result.get("Relevance") == "YES":
            output_data = True
        else:
            output_data = False

        return ChainResult(
            chain_name=self.name,
            input_data="\n".join([doc.page_content for doc in docs]),
            output_data=output_data,
            success=True,
        )


class HTMLRelevanceCondition(BoolRoutingCondition):
    def __init__(self, llm: ChatAdapter):
        super().__init__()
        self.chain = HTMLRelevance(llm=llm)

    def evaluate(self, docs: list[Document]) -> bool:
        result = self.chain.invoke(docs=docs)
        if not result.success:
            raise Exception(
                f"Error evaluating HTMLRelevanceCondition: {result.error_message}"
            )
        return result.output_data


class HTMLBlockExtraction(BaseChain):
    def __init__(
        self,
        llm: ChatAdapter,
        evaluators: list[BaseEvaluator] | None = None,
        eval_max_retries: int = 0,
        invoke_max_retries: int = 0,
    ):
        super().__init__(
            evaluators=evaluators,
            eval_max_retries=eval_max_retries,
            invoke_max_retries=invoke_max_retries,
        )
        self.llm = llm

    @staticmethod
    def plant_unit_tags(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        # The complete list of block-level elements
        target_tags = [
            "p",
            "div",
            "table",
            "ul",
            "ol",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "blockquote",
        ]

        chunk_counter = 0

        for tag in soup.find_all(target_tags):
            # Skip empty tags to save LLM tokens
            if not tag.get_text(strip=True):
                continue

            # Inject the unique ID
            tag["id"] = f"chunk-{chunk_counter}"
            chunk_counter += 1

        # Convert back to a string to send to the LLM
        html_for_llm = str(soup)

        return html_for_llm

    def _invoke(self, docs: list[Document]) -> ChainResult:
        system_prompt = """
        You are an expert web scraping routing assistant. Your job is to analyze the provided HTML layout
        and identify the exact data-chunk-id of the container element that holds the outstanding common shares of the company.

        Outstanding common shares refers to the total number of shares AFTER any IPO or corporate actions.

        - Share offered in an IPO should NOT be counted as outstanding shares.
        - If there are multiple classes of shares, only extract the number of outstanding common class A shares.
        """

        if len(docs) != 1:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=f"Expected exactly 1 document as input, but got {len(docs)}",
            )

        soup = BeautifulSoup(docs[0].page_content, "html.parser")
        # The complete list of block-level elements
        target_tags = [
            "p",
            "div",
            "table",
            "ul",
            "ol",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "blockquote",
        ]

        chunk_counter = 0
        saved_chunks = {}

        for tag in soup.find_all(target_tags):
            # Skip empty tags to save LLM tokens
            text_content = tag.get_text(separator=" ", strip=True).replace("\xa0", " ")
            if not text_content:
                continue
            parent_tag = tag.find_parent(target_tags)
            parent_id = parent_tag.get("id") if parent_tag else None
            # Inject the unique ID
            chunk_id = f"chunk-{chunk_counter}"
            tag["id"] = chunk_id
            saved_chunks[chunk_id] = {
                "tag_type": tag.name,
                "text": text_content,  # The clean, readable text
                "parent_id": parent_id,
                "html": str(tag),  # The exact HTML (crucial if it's a table!)
            }
            chunk_counter += 1

        # Convert back to a string to send to the LLM
        html_for_llm = str(soup)

        all_fields = {
            "justification": (
                str,
                pydantic.Field(description="Short semantic reason why the choice."),
            ),
            "chosen_tag": (Literal[*list(saved_chunks.keys())], ...),
        }

        ResponseModel = pydantic.create_model("ResponseModel", **all_fields)

        # extracted_data_chunk_id = {"data_chunk_id": (str, ...)}

        # ResponseModel = pydantic.create_model("ResponseModel", **extracted_data_chunk_id)

        payload = {
            "think": False,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": docs[0].page_content},
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0},
        }

        tag_result = self.llm.chat_with_expected_type_result(
            chat_payload=payload, result_type="dict"
        )

        if not tag_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=tag_result.error_message,
            )

        if tag_result.model_result.get("chosen_tag") not in saved_chunks:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=f"Model returned a chosen_tag that does not exist in the HTML: '{tag_result.model_result.get('chosen_tag')}'",
            )

        relevant_tag = saved_chunks.get(tag_result.model_result.get("chosen_tag"))
        if relevant_tag:
            print(
                f"[{tag_result.model_result.get('chosen_tag')}] Type: <{relevant_tag['tag_type']}>"
            )

            # If it's a table, we probably want the HTML so we can parse it with Pandas
            if relevant_tag["tag_type"] == "table":
                print(f"Table HTML: {relevant_tag['html']}")
                output_data = relevant_tag["html"]

            # If it's a paragraph, we usually just want the text
            else:
                print(f"Text Content: {relevant_tag['text']}")
                output_data = relevant_tag["text"]

            print("-" * 40)

        return ChainResult(
            chain_name=self.name,
            input_data="\n".join([doc.page_content for doc in docs]),
            output_data=output_data,
            success=True,
            metadata={"full_html_block": docs[0].page_content},
        )


class HTMLBlockQuality(BaseEvaluator):

    trigger_retry: bool = True
    hard_fail: bool = False

    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:

        full_html = output.metadata.get("full_html_block", "")
        soup = BeautifulSoup(full_html, "html.parser")
        full_page_text = soup.get_text(separator=" ", strip=True)
        total_page_length = len(full_page_text)

        block_length = len(output.output_data.strip())

        if total_page_length == 0:
            return EvalResult(
                eval_name=self.name,
                eval_success=True,
                passed=True,
                feedback="Warning: The full page text length is 0, cannot evaluate block quality.",
            )

        ratio = block_length / total_page_length
        print(
            f"📊 Extraction Ratio: {ratio:.1%} ({block_length}/{total_page_length} chars)"
        )

        if ratio > 0.8:
            print(
                "❌ INVALID: The model lazily output a text block holding almost the whole page."
            )
            return EvalResult(
                eval_name=self.name,
                eval_success=True,
                passed=False,
                error_message=f"Extracted block is {ratio:.1%} of the page, which is too large and likely not specific.",
            )

        return EvalResult(eval_name=self.name, eval_success=True, passed=True)


class HTMLExtraction(BaseChain):
    def __init__(
        self,
        llm: ChatAdapter,
        evaluators: list[BaseEvaluator] | None = None,
        eval_max_retries: int = 0,
        invoke_max_retries: int = 0,
    ):
        super().__init__(
            evaluators=evaluators,
            eval_max_retries=eval_max_retries,
            invoke_max_retries=invoke_max_retries,
        )
        self.llm = llm

    def _invoke(self, docs: list[Document]) -> ChainResult:
        system_prompt = """
        You are an expert web scraping routing assistant. Your job is to analyze the provided HTML layout
        and identify the the outstanding common shares of the company.

        Outstanding common shares refers to the total number of shares AFTER any IPO or corporate actions.
        """

        if len(docs) != 1:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=f"Expected exactly 1 document as input, but got {len(docs)}",
            )

        all_fields = {
            "justification": (
                str,
                pydantic.Field(description="Short semantic reason why the choice."),
            ),
            "outstanding_common_shares": (float, ...),
        }

        ResponseModel = pydantic.create_model("ResponseModel", **all_fields)

        payload = {
            "think": False,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": docs[0].page_content},
            ],
            "format": ResponseModel.model_json_schema(),
            "options": {"temperature": 0},
        }

        extraction_result = self.llm.chat_with_expected_type_result(
            chat_payload=payload, result_type="dict"
        )

        if not extraction_result.success:
            return ChainResult(
                chain_name=self.name,
                input_data="\n".join([doc.page_content for doc in docs]),
                output_data=None,
                success=False,
                error_message=extraction_result.error_message,
            )

        outstanding_common_shares = extraction_result.model_result.get(
            "outstanding_common_shares"
        )

        return ChainResult(
            chain_name=self.name,
            input_data="\n".join([doc.page_content for doc in docs]),
            output_data=outstanding_common_shares,
            success=True,
            metadata={"full_html_block": docs[0].page_content},
        )


class CheckFloatinSource(BaseEvaluator):

    trigger_retry: bool = True
    hard_fail: bool = True

    def _evaluate(self, inputs: list[Document], output: ChainResult) -> EvalResult:

        full_html = output.metadata.get("full_html_block", "")
        clean_text = full_html.replace("\xa0", " ")

        # Regex groups: (The Number) (Optional Scale Word)
        # \s* handles any spaces between the number and the word
        pattern = r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|thousand|trillion|m|b|k)?\b"

        # Ignore case so "Million" and "million" both work
        matches = re.findall(pattern, clean_text, re.IGNORECASE)

        scale_multipliers = {
            "thousand": 1_000,
            "k": 1_000,
            "million": 1_000_000,
            "m": 1_000_000,
            "billion": 1_000_000_000,
            "b": 1_000_000_000,
            "trillion": 1_000_000_000_000,
        }

        for num_str, scale_word in matches:
            clean_num_str = num_str.replace(",", "")

            try:
                parsed_float = float(clean_num_str)

                # Apply the multiplier if a scale word was found
                if scale_word:
                    scale_word = scale_word.lower()
                    parsed_float *= scale_multipliers.get(scale_word, 1)

                if math.isclose(parsed_float, output.output_data, rel_tol=1e-5):
                    return EvalResult(
                        eval_name=self.name, eval_success=True, passed=True
                    )

            except ValueError:
                continue

        return EvalResult(
            eval_name=self.name,
            eval_success=True,
            passed=False,
            error_message="Could not find a float in the source text that matches the extracted value.",
        )


HTMLBlockExtractionChain = HTMLBlockExtraction(
    llm=ollamaChat(model="gemma4:e4b-it-qat"),
    evaluators=[HTMLBlockQuality()],
    eval_max_retries=5,
    invoke_max_retries=1,
)


HTMLExtractionChain = HTMLExtraction(
    llm=ollamaChat(model="gemma4:e4b-it-qat"),
    evaluators=[CheckFloatinSource()],
    eval_max_retries=5,
    invoke_max_retries=1,
)

html_relevance_route = BoolRoute(
    name="HTMLRelevance",
    condition=HTMLRelevanceCondition(llm=ollamaChat(model="gemma4:e4b-it-qat")),
    destination_chain=HTMLExtractionChain,
)


process_chain = BoolRouterChain(
    routes=[html_relevance_route], custom_name="HTMLBlockExtractionRouter"
)


tracer = trace.get_tracer(__name__)


def process_all_markdowns():
    all_results = []
    for each_file in glob.glob("sec_filings/ingest/*.html"):
        with tracer.start_as_current_span("html.file") as span:
            span.set_attribute("html.file.name", each_file)
            print(f"Processing file: {each_file}")
            with open(each_file, "r") as f:
                content = f.read()

            docs = [Document(page_content=content)]
            splitter = HTMLSplitter(chunk_size_words=200, overlap_words=50)
            split_docs = splitter.split_documents(docs)
            for each_doc in split_docs:

                final_result = process_chain.invoke(docs=[each_doc])

                if final_result.success and final_result.output_data:
                    print(f"final_result = {final_result}")
                    print(final_result.to_dict())
                    all_results.append(final_result.to_dict())

            with open("sec_filings/results.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, default=str)


if __name__ == "__main__":
    # Initialize tracing before starting the main process
    # Traces will be saved to logs/otel_traces.log
    setup_otel_tracing(log_dir=os.path.join("sec_filings", "traces"))

    process_all_markdowns()
