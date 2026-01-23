"""
Completion Logger - Saves model-generated completions to JSON files for inspection.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class CompletionLogger:
    """Logger that saves model completions to JSON files."""
    
    _instance: Optional["CompletionLogger"] = None
    
    def __init__(
        self,
        output_dir: str,
        enabled: bool = True,
        max_samples_per_file: int = 100,
    ):
        """
        Initialize the completion logger.
        
        Args:
            output_dir: Directory to save log files
            enabled: Whether logging is enabled
            max_samples_per_file: Maximum samples before creating a new file
        """
        self.output_dir = output_dir
        self.enabled = enabled
        self.max_samples_per_file = max_samples_per_file
        self.samples: List[Dict[str, Any]] = []
        self.file_counter = 0
        self.total_logged = 0
        
        if self.enabled:
            self.log_dir = os.path.join(output_dir, "completion_logs")
            os.makedirs(self.log_dir, exist_ok=True)
    
    @classmethod
    def get_instance(cls) -> Optional["CompletionLogger"]:
        """Get the singleton instance."""
        return cls._instance
    
    @classmethod
    def initialize(
        cls,
        output_dir: str,
        enabled: bool = True,
        max_samples_per_file: int = 100,
    ) -> "CompletionLogger":
        """Initialize and set the singleton instance."""
        cls._instance = cls(
            output_dir=output_dir,
            enabled=enabled,
            max_samples_per_file=max_samples_per_file,
        )
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        if cls._instance is not None:
            cls._instance.flush()
        cls._instance = None
    
    def log_completion(
        self,
        entry_point: str,
        aux_completion: str,
        main_completion: str,
        aux_cleaned: str = "",
        main_cleaned: str = "",
        aux_extracted: str = "",
        main_extracted: str = "",
        reward: float = 0.0,
        level1_reward: float = 0.0,
        level2_reward: float = 0.0,
        level3_reward: float = 0.0,
        passed_tests: int = 0,
        total_tests: int = 0,
        phase: str = "train",
        turn: int = 1,
        step: int = 0,
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a single completion pair.
        
        Args:
            entry_point: The function name being tested
            aux_completion: Raw aux function completion
            main_completion: Raw main function completion
            aux_cleaned: Cleaned aux completion
            main_cleaned: Cleaned main completion
            aux_extracted: Extracted aux function
            main_extracted: Extracted main function
            reward: Total reward
            level1_reward: Level 1 reward
            level2_reward: Level 2 reward
            level3_reward: Level 3 reward
            passed_tests: Number of passed tests
            total_tests: Total number of tests
            phase: "train" or "eval"
            turn: Turn number (for multi-turn)
            step: Training step
            extra_info: Additional information to log
        """
        if not self.enabled:
            return
        
        sample = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "phase": phase,
            "turn": turn,
            "entry_point": entry_point,
            "completions": {
                "aux_raw": aux_completion,
                "main_raw": main_completion,
                "aux_cleaned": aux_cleaned,
                "main_cleaned": main_cleaned,
                "aux_extracted": aux_extracted,
                "main_extracted": main_extracted,
            },
            "rewards": {
                "total": reward,
                "level1": level1_reward,
                "level2": level2_reward,
                "level3": level3_reward,
            },
            "test_results": {
                "passed": passed_tests,
                "total": total_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            },
        }
        
        if extra_info:
            sample["extra"] = extra_info
        
        self.samples.append(sample)
        self.total_logged += 1
        
        if len(self.samples) >= self.max_samples_per_file:
            self.flush()
    
    def flush(self):
        """Write buffered samples to file."""
        if not self.enabled or not self.samples:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"completions_{timestamp}_{self.file_counter:04d}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.samples, f, indent=2, ensure_ascii=False)
        
        self.samples = []
        self.file_counter += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_logged": self.total_logged,
            "files_written": self.file_counter,
            "pending_samples": len(self.samples),
            "log_dir": self.log_dir if self.enabled else None,
        }


def log_completion(
    entry_point: str,
    aux_completion: str,
    main_completion: str,
    **kwargs,
):
    """
    Convenience function to log a completion using the singleton logger.
    Does nothing if logger is not initialized.
    """
    logger = CompletionLogger.get_instance()
    if logger is not None:
        logger.log_completion(
            entry_point=entry_point,
            aux_completion=aux_completion,
            main_completion=main_completion,
            **kwargs,
        )
