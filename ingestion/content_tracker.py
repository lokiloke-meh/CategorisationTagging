import os
import json
import re
import hashlib
from datetime import datetime


class ContentTracker:
    """
    A reusable class for tracking scraped file contents to prevent duplicate downloads.
    It maps SHA-256 hashes of file contents (or cleaned text) to their saved file paths.
    """
    def __init__(self, hash_map_filepath: str, raw_files_directory: str):
        """
        Args:
            hash_map_filepath (str): Path to the JSON file where hash mappings are saved.
            raw_files_directory (str): The root folder to verify if the physical files still exist.
        """
        self.hash_map_filepath = hash_map_filepath
        self.raw_files_directory = raw_files_directory
        self.hashes = self._load()

    def _load(self) -> dict:
        """Internal method to load the hash map from disk."""
        if os.path.exists(self.hash_map_filepath):
            with open(self.hash_map_filepath, 'r', encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self):
        """Internal method to persist the hash map to disk."""
        with open(self.hash_map_filepath, 'w', encoding="utf-8") as f:
            json.dump(self.hashes, f, indent=4)

    def get_text_hash(self, text: str, ignore_pattern: str = None) -> str:
        """
        Hashes text content. Optionally strips dynamic content (like timestamps) 
        before hashing so identical pages scraped on different days match perfectly.
        """
        if ignore_pattern:
            text = re.sub(ignore_pattern, '', text)
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get_bytes_hash(self, data: bytes) -> str:
        """Hashes raw binary data (e.g., PDF file bytes)."""
        return hashlib.sha256(data).hexdigest()

    def check_duplicate(self, file_hash: str, min_timestamp: str = None) -> str:
        """
        Checks if the file hash is known AND if the file physically exists on the drive.
        If min_timestamp is provided, records older than the timestamp are ignored.
        Returns the relative filepath if it's a valid duplicate, else Returns None.
        """
        if file_hash in self.hashes:
            entry = self.hashes[file_hash]
            
            # Handle legacy string format vs new dictionary format
            if isinstance(entry, dict):
                rel_path = entry.get("path")
                ts = entry.get("processed_timestamp")
            else:
                rel_path = entry
                ts = None

            # Staleness check
            if min_timestamp and (not ts or ts < min_timestamp):
                return None

            full_path = os.path.join(self.raw_files_directory, rel_path)
            if os.path.exists(full_path):
                return rel_path
        return None

    def mark_saved(self, file_hash: str, relative_path: str, source_url: str = None, title: str = None):
        """Registers a newly saved file's hash, path, timestamp, source URL, and title."""
        self.hashes[file_hash] = {
            "path": relative_path,
            "processed_timestamp": datetime.now().isoformat(),
            "source_url": source_url,
            "title": title
        }
        self._save()


class ContentReader:
    """
    A utility class to fetch content from the disk using hashes 
    verified by a ContentTracker instance.
    """
    def __init__(self, tracker: ContentTracker):
        self.tracker = tracker

    def fetch_markdown_content(self, file_hash: str) -> str | None:
        """
        Retrieves the string content of a markdown file associated with the given hash.
        Returns None if the hash is unknown, the file is missing, or not a markdown file.
        """
        relative_path = self.tracker.check_duplicate(file_hash)
        
        if relative_path and relative_path.lower().endswith('.md'):
            full_path = os.path.join(self.tracker.raw_files_directory, relative_path)
            try:
                with open(full_path, 'r', encoding="utf-8") as f:
                    return f.read()
            except Exception:
                # Returns None if the file cannot be read (e.g., permissions or deleted since check)
                return None
        return None