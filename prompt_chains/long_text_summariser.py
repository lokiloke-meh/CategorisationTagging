import re
from abc import ABC, abstractmethod
from typing import Optional
from utils.myOllama import AIChat, ollamaChat
from prompt_chains.base_chains import Document, BaseChain, ChainResult
from evals.base_evals import BaseEvaluator
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

# ==========================================
# Text Splitter Classes
# ==========================================
class TextSplitter(ABC):
    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def split_documents(self):
        pass



class WordTextSplitter(TextSplitter):
    """Splits text rigidly by word count."""
    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 100):
        super(WordTextSplitter, self).__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    def split_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunked_docs = []
        for doc in documents:
            for chunk in self.split_text(doc.page_content):
                chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        return chunked_docs



class SentenceTextSplitter(TextSplitter):
    """Splits text by sentence boundaries to ensure thoughts aren't cut in half."""
    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 100):
        super(SentenceTextSplitter, self).__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> list[str]:
        # Split by punctuation followed by a space (e.g., ". ", "! ", "? ")
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        i = 0
        
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())
            
            # If adding this sentence stays under the limit, or the chunk is empty
            if current_word_count + sentence_words <= self.chunk_size or not current_chunk:
                current_chunk.append(sentence)
                current_word_count += sentence_words
                i += 1
            else:
                # Chunk is full, save it
                chunks.append(" ".join(current_chunk))
                
                # Create the overlap for the next chunk
                overlap_chunk = []
                overlap_word_count = 0
                # Traverse backwards through the current chunk to gather overlap sentences
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_word_count + s_words <= self.chunk_overlap:
                        overlap_chunk.insert(0, s) # Add to front to maintain order
                        overlap_word_count += s_words
                    else:
                        break
                        
                # Start the next chunk with the overlap
                current_chunk = overlap_chunk
                current_word_count = overlap_word_count

        # Append the final chunk if anything is left
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunked_docs = []
        for doc in documents:
            for chunk in self.split_text(doc.page_content):
                chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata, title=doc.title))
        return chunked_docs


# ==========================================
# Prompt Chain Classes
# ==========================================

class RefineSummarizeChain(BaseChain):
    """Processes chunks sequentially, updating the summary as it goes."""
    def __init__(self, llm: AIChat, 
                 text_splitter: Optional[TextSplitter] = None,
                 evaluators: list[BaseEvaluator] = None):
        super().__init__(evaluators=evaluators)
        self.llm = llm
        self.text_splitter = text_splitter


    def _invoke(self, docs: list[Document]) -> ChainResult:
        chain_name = self.__class__.__name__
        if self.text_splitter:
            docs = self.text_splitter.split_documents(docs)

        if not docs: 
            result = ChainResult(chain_name=chain_name,
                                 input_data="", 
                                 output_data="",
                                 success=False, 
                                 error_message="Empty docs passed")
            return result

        print(f"--- REFINE CHAIN: Processing {len(docs)} chunks ---\n")
        
        current_summary = None
        for i, doc in enumerate(docs):
            print(f" > Processing chunk {i + 1}/{len(docs)}...")
            if i == 0:
                system_prompt = f"""
                You are an assistant analyzing parts of a large document. 
                Read the text given by the user and extract all relevant information and key entities.
                Only use the dates explicitly stated in the provided document. If a year, date, or number is not explicitly mentioned, do not mention it instead of guessing.

                Rules:
                Be objective, concise, thorough and accurate.
                Do not add conversational fillers or summaries outside the extraction.
                Your summary should not exceed 2000 characters.
                """
            else:

                previous_summary = current_summary.model_result

                system_prompt = f"""
                Based on an existing summary of a document:

                EXISTING SUMMARY:
                {previous_summary}

                Your goal is to update the current summary to incorporate the new information from a new section of text given by the user.
                If the new section provides more detail on existing themes, expand the summary. 
                If the new section introduces new topics, add them logically.
                If the new context does not add significant impactful information, return the original summary unchanged.

                Rules:
                Be objective, concise, thorough and accurate.
                Do not add conversational fillers or summaries outside the extraction.
                Prioritise the overarching topic.
                Your summary should not exceed 2000 characters.
                """

            payload = {
                "think": False, 
                "stream": False,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": doc.page_content}],
                "options": {"temperature": 0}
            }

            current_summary = self.llm.chat_with_expected_type_result(chat_payload=payload, result_type="string")

            if not current_summary.success:
                result = ChainResult(chain_name=chain_name,
                                     input_data=doc.page_content,
                                     output_data="",
                                     success=False,
                                     error_message=current_summary.error_message
                                     )
                return result

        # Prepend unique document titles to provide context to the summary
        unique_titles = sorted({doc.title for doc in docs if doc.title})
        if unique_titles:
            current_summary.model_result = f"Title: {'. '.join(unique_titles)}. {current_summary.model_result}"

        return ChainResult(chain_name=chain_name,
                           input_data="\n".join([doc.page_content for doc in docs]),
                           output_data=current_summary.model_result,
                           success=True
                           )



class MapReduceSummarizeChain(BaseChain):
    """Summarizes all chunks individually, then summarizes the summaries."""
    def __init__(self, llm: AIChat, 
                 text_splitter: Optional[TextSplitter] = None,
                 evaluators: list[BaseEvaluator] = None):
        super().__init__(evaluators=evaluators)
        self.llm = llm
        self.text_splitter = text_splitter


    map_phase_system_prompt = f"""
    You are an assistant analyzing parts of a large document. 
    Read the text given by the user and extract all relevant information and key entities.
    Only use the dates explicitly stated in the provided document. If a year, date, or number is not explicitly mentioned, do not mention it instead of guessing.

    Rules:
    Be objective, concise, thorough and accurate.
    Do not add conversational fillers or summaries outside the extraction.
    Your summary should not exceed 4000 characters.
    """

    reduce_phase_system_prompt = f"""
    You are an assistant analyzing a large document. 
    Based on the following partial summaries extracted from different parts of the document given by the user, distill them into a final, comprehensive, and cohesive master summary.
    Only use the dates explicitly stated in the provided document. If a year, date, or number is not explicitly mentioned, do not mention it instead of guessing.

    Rules:
    Be objective, concise and coherent.
    The aggregate summary must cover all key points providing relevant details and key entities, while avoiding any unnecessary information or repetition.
    Your summary should not exceed 4000 characters.
    """


    def _invoke(self, docs: list[Document]) -> ChainResult:
        chain_name = self.__class__.__name__
        if self.text_splitter:
            docs = self.text_splitter.split_documents(docs)

        if not docs: 
            result = ChainResult(chain_name=chain_name,
                                 input_data="", 
                                 output_data="",
                                 success=False, 
                                 error_message="Empty docs passed")
            return result
        
        # Pre-extract unique titles and create a formatted header string
        unique_titles = sorted({doc.title for doc in docs if doc.title})
        formatted_titles = f"Title: {'. '.join(unique_titles)}. " if unique_titles else ""
        
        # MAP PHASE
        print(f"--- MAP PHASE: Processing {len(docs)} individual chunks ---\n")
        chunk_summaries = []
        for i, doc in enumerate(docs):
            print(f" > Mapping summary for chunk {i + 1}/{len(docs)}...")

            payload = {
                "think": False, 
                "stream": False,
                "messages": [{"role": "system", "content": self.__class__.map_phase_system_prompt},
                             {"role": "user", "content": doc.page_content}],
                "options": {"temperature": 0}
            }
            partial_summary = self.llm.chat_with_expected_type_result(chat_payload=payload, result_type="string")

            if not partial_summary.success:
                result = ChainResult(chain_name=chain_name,
                                     input_data=doc.page_content, 
                                     output_data="",
                                     success=False, 
                                     error_message=partial_summary.error_message)
                return result
            
            chunk_summaries.append(partial_summary.model_result)


        # no point going to reduce phase if only 1 doc
        if len(docs) == 1:
            if formatted_titles:
                partial_summary.model_result = formatted_titles + partial_summary.model_result

            return ChainResult(chain_name=chain_name,
                               input_data="\n".join([doc.page_content for doc in docs]),
                               output_data=partial_summary.model_result,
                               success=True
                               )
        

        # REDUCE PHASE
        print("\n--- REDUCE PHASE: Combining chunk summaries ---\n")

        if unique_titles:
            chunk_summaries = unique_titles + chunk_summaries

        combined_summaries_text = "\n\n---\n\n".join(chunk_summaries)
        payload = {
            "think": False, 
            "stream": False,
            "messages": [{"role": "system", "content": self.__class__.reduce_phase_system_prompt},
                         {"role": "user", "content": combined_summaries_text}],
            "options": {"temperature": 0}
        }
        aggregate_summary = self.llm.chat_with_expected_type_result(chat_payload=payload, result_type="string")
        if not aggregate_summary.success:
            result = ChainResult(chain_name=chain_name,
                                 input_data=combined_summaries_text, 
                                 output_data="",
                                 success=False, 
                                 error_message=aggregate_summary.error_message)
            return result
        
        if formatted_titles:
            aggregate_summary.model_result = formatted_titles + aggregate_summary.model_result

        return ChainResult(chain_name=chain_name,
                           input_data="\n".join([doc.page_content for doc in docs]),
                           output_data=aggregate_summary.model_result,
                           success=True
                           )
