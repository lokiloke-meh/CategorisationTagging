from bs4 import BeautifulSoup, NavigableString, Tag
from prompt_chains.base_chains import Document


class HTMLSplitter:
    """
    Splits an HTML document into chunks by keeping logical block elements intact 
    (e.g., paragraphs, tables, lists). Removes scripts and styles automatically.
    """
    def __init__(self, chunk_size_words: int = 500, overlap_words: int = 100):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        
        # Elements that represent self-contained logical blocks of text
        self.block_tags = {
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
            'ul', 'ol', 'table', 'pre', 'blockquote', 'article', 'section'
        }

    def _extract_blocks(self, node) -> list[str]:
        """Recursively walks the DOM to extract intact block elements."""
        blocks = []
        for child in node.children:
            # Base Text Node (e.g., floating text directly inside a div)
            if isinstance(child, NavigableString):
                text = child.strip()
                if text:
                    # Wrap orphaned text in a generic div so it stays valid HTML
                    blocks.append(f"<div>{text}</div>")
            
            # HTML Tag
            elif isinstance(child, Tag):
                if child.name in self.block_tags:
                    # If it's a logical block (like a table), grab the WHOLE thing intact!
                    blocks.append(str(child))
                else:
                    # If it's a layout tag (like <div>, <main>, <body>), dive deeper into its children
                    blocks.extend(self._extract_blocks(child))
        return blocks

    def _count_words(self, html_str: str) -> int:
        """Counts only the actual visible text words, ignoring HTML syntax."""
        return len(BeautifulSoup(html_str, "html.parser").get_text().split())

    def split_text(self, html_text: str) -> list[str]:
        soup = BeautifulSoup(html_text, "html.parser")
        
        # PRE-PROCESSING: Destroy scripts, styles, and SVGs to save LLM context tokens!
        for useless_tag in soup(["script", "style", "meta", "noscript", "svg", "nav", "footer"]):
            useless_tag.decompose()

        # 1. Flatten the DOM into an array of intact HTML blocks
        blocks = self._extract_blocks(soup)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        i = 0
        while i < len(blocks):
            block = blocks[i]
            block_words = self._count_words(block)
            
            # Edge Case: A single block (like a massive table) is larger than our chunk size.
            # We yield it as an oversized chunk rather than breaking the HTML table syntax.
            if block_words > self.chunk_size_words:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                chunks.append(block)
                i += 1
                continue
                
            # If adding this block keeps us under the limit, append it.
            if current_word_count + block_words <= self.chunk_size_words:
                current_chunk.append(block)
                current_word_count += block_words
                i += 1
            else:
                # Chunk is full! Join the blocks and save.
                chunks.append("\n".join(current_chunk))
                
                # Calculate the overlap backwards to maintain context
                overlap_chunk = []
                overlap_word_count = 0
                for b in reversed(current_chunk):
                    b_words = self._count_words(b)
                    if overlap_word_count + b_words <= self.overlap_words:
                        overlap_chunk.insert(0, b) # Insert at front to maintain chronological order
                        overlap_word_count += b_words
                    else:
                        break

                # Keep overlap only while it still lets the next block make progress.
                while overlap_chunk and overlap_word_count + block_words > self.chunk_size_words:
                    removed_block = overlap_chunk.pop(0)
                    overlap_word_count -= self._count_words(removed_block)
                
                # Start the next chunk with our overlapped blocks
                current_chunk = overlap_chunk
                current_word_count = overlap_word_count
                
        # Append whatever is left over
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunked_docs = []
        for doc in documents:
            for i, chunk in enumerate(self.split_text(doc.page_content)):
                # We inject 'chunk_index' metadata so downstream chains know the chronological order
                meta = doc.metadata.copy()
                meta["html_chunk_index"] = i + 1
                chunked_docs.append(Document(page_content=chunk, metadata=meta))
        return chunked_docs


