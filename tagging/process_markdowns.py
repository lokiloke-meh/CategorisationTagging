import os
import json
import hashlib
from typing import Any
import sys
from datetime import datetime
from utils.myOllama import LLMResult, ollamaChat



class ProcessingCache:
    """
    Tracks the execution of custom functions on files.
    Ensures a function is only re-run if the file's content changes 
    OR if the parameters passed to the function change.
    """
    def __init__(self, cache_filepath: str, min_timestamp: str = None):
        self.cache_filepath = cache_filepath
        self.cache = self._load(min_timestamp)

    def _load(self, min_timestamp: str = None) -> dict:
        if os.path.exists(self.cache_filepath):
            with open(self.cache_filepath, 'r', encoding="utf-8") as f:
                data = json.load(f)
            
            if not min_timestamp:
                return data

            # Only keep entries that are newer than or equal to the min_timestamp
            return {k: v for k, v in data.items() if v.get("processed_timestamp", "") >= min_timestamp}
        return {}

    def _save(self):
        with open(self.cache_filepath, 'w', encoding="utf-8") as f:
            # Added default=str as a safety fallback for any non-serializable 
            # objects that might sneak into metadata or error messages.
            json.dump(self.cache, f, indent=4, default=str)

    def _hash_parameter(self, param: Any) -> str:
        """
        Creates a deterministic hash of the parameter.
        Sorts JSON keys so {"A": 1, "B": 2} and {"B": 2, "A": 1} result in the same hash.
        """
        param_str = json.dumps(param, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode('utf-8')).hexdigest()

    def get_cached_result(self, file_hash: str, param: Any, ignore_failures: bool = False, min_timestamp: str = None) -> Any:
        """
        Retrieves the result IF the file hash exists AND the parameter hasn't changed.
        If ignore_failures is True, cached results that marked a failure (success=False) 
        are treated as cache misses to trigger a re-run.
        If min_timestamp is provided (ISO string), results older than this are ignored.
        Returns None if a re-run is required.
        """
        param_hash = self._hash_parameter(param)
        
        if file_hash in self.cache:
            entry = self.cache[file_hash]
            # Check if the parameter used last time perfectly matches the current parameter
            if entry.get("param_hash") == param_hash:
                
                # Check if the record is too old
                if min_timestamp:
                    cached_ts = entry.get("processed_timestamp")
                    # If no timestamp or older than min_timestamp, invalidate
                    if not cached_ts or cached_ts < min_timestamp:
                        return None

                result_data = entry.get("result")
                
                # Logic to selectively ignore failed entries
                if ignore_failures:
                    is_success = True
                    if isinstance(result_data, dict):
                        is_success = result_data.get("success", True)
                    
                    if not is_success:
                        return None

                # Reconstruct LLMResult if the dictionary matches the LLMResult structure
                if isinstance(result_data, dict) and "model_name" in result_data and "success" in result_data:
                    return LLMResult(**result_data)
                return result_data
                
        return None

    def save_result(self, file_hash: str, param: Any, result: Any):
        """Saves the result and the parameter state to the cache."""
        # Convert dataclasses to dicts using our refactored to_dict() method
        serializable_result = result.to_dict() if hasattr(result, "to_dict") else result

        self.cache[file_hash] = {
            "processed_timestamp": datetime.now().isoformat(),
            "param_hash": self._hash_parameter(param),
            "parameter_readable": param,  # Kept in plain text so you can read the JSON manually
            "result": serializable_result
        }
        self._save()


class CacheQuery:
    """
    Utility class to perform searches and filtering on a ProcessingCache instance.
    """
    def __init__(self, processor: ProcessingCache):
        self.processor = processor

    def query(self, predicate) -> list[str]:
        """
        Returns a list of file hashes where predicate(result_dict) is True.
        """
        return [
            file_hash for file_hash, entry in self.processor.cache.items()
            if entry.get("result") and predicate(entry["result"])
        ]

    def find_by_tags_score(self, tag_name: str, target_levels: list[str] = None) -> list[str]:
        """
        Helper to find hashes where a specific tag level in output_data matches the target levels.
        Default target_levels is ["High"].
        """
        if target_levels is None:
            target_levels = ["High"]

        def check_score(res):
            # Safely navigate the result structure, handling potential None values
            output_data = res.get("output_data") or {}
            tags = output_data.get("tags") or {}
            score = tags.get(tag_name)
            # Ensure the score is not None and exists within the target levels
            return score is not None and score in target_levels

        return self.query(check_score)

    def get_full_cached_entry(self, file_hash: str) -> dict | None:
        """
        Retrieves the full cached entry (including metadata, parameters, and result)
        for a given file hash.
        Returns None if the hash is not found in the cache.
        """
        return self.processor.cache.get(file_hash)
