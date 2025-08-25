from typing import Any, Dict, List, Optional

import numpy as np


def mt_humaneval_logger(
    completions1_turns: List[List[str]],
    completions2_turns: List[List[str]],
    test_cases: List[str],
    entry_points: List[str],
    prompts: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Multi-turn logger for code generation tasks (HumanEval/CoopHumanEval) with aux + main function collaboration.

    Args:
        completions1_turns: List of lists - [sample][turn] completions for agent 1
        completions2_turns: List of lists - [sample][turn] completions for agent 2
        test_cases: List of test cases
        entry_points: List of entry point function names
        prompts: Optional list of prompts for import extraction

    Returns:
        List of metric dictionaries with multi-turn information
    """
    from experiments.loggers.code_logger import code_reward_logger

    all_metrics = []

    # Get number of turns from the data
    num_turns = (
        len(completions1_turns[0])
        if completions1_turns and completions1_turns[0]
        else 0
    )

    for i in range(len(test_cases)):
        sample_metrics = {
            "sample_id": i,
            "entry_point": entry_points[i],
            "num_turns": 0,
            "early_termination": False,
            "termination_turn": -1,
        }

        # Process each turn
        turn_rewards = []
        early_terminated = False

        for turn_idx in range(num_turns):
            if early_terminated:
                # Fill with max values for remaining turns
                turn_rewards.append(4.0)

                # Add perfect metrics for this turn
                sample_metrics[f"turn_{turn_idx + 1}/total_reward"] = 4.0
                sample_metrics[f"turn_{turn_idx + 1}/gated_total_reward"] = 4.0
                sample_metrics[f"turn_{turn_idx + 1}/level_1_reward"] = 1.0
                sample_metrics[f"turn_{turn_idx + 1}/level_2_reward"] = 0.5
                sample_metrics[f"turn_{turn_idx + 1}/level_3_reward"] = 2.5
                sample_metrics[f"turn_{turn_idx + 1}/test_reward"] = 1.0
                sample_metrics[f"turn_{turn_idx + 1}/passed_tests"] = (
                    sample_metrics.get(f"turn_1/total_tests", 0)
                )
                sample_metrics[f"turn_{turn_idx + 1}/total_tests"] = sample_metrics.get(
                    f"turn_1/total_tests", 0
                )
                sample_metrics[f"turn_{turn_idx + 1}/passed_rate"] = 1.0
                sample_metrics[f"turn_{turn_idx + 1}/timeout_num"] = 0
                sample_metrics[f"turn_{turn_idx + 1}/bonus_reward"] = 1.5  # 0.5 + 1.0
                sample_metrics[f"turn_{turn_idx + 1}/aux_usage_bonus"] = 0.5
                sample_metrics[f"turn_{turn_idx + 1}/anti_wrapper_bonus"] = 1.0
                sample_metrics[f"turn_{turn_idx + 1}/called_wo_used_deduction"] = 0.0
                sample_metrics[f"turn_{turn_idx + 1}/early_termination_filled"] = 1

                # Improvement is 0 since both turns have perfect score
                if turn_idx > 0:
                    sample_metrics[
                        f"turn_{turn_idx + 1}/improvement_from_turn_{turn_idx}"
                    ] = 0.0

                sample_metrics["num_turns"] = turn_idx + 1
                continue

            # Extract completions for this turn
            turn_completions1 = [
                (
                    completions1_turns[j][turn_idx]
                    if j < len(completions1_turns)
                    and turn_idx < len(completions1_turns[j])
                    else ""
                )
                for j in range(len(test_cases))
            ]
            turn_completions2 = [
                (
                    completions2_turns[j][turn_idx]
                    if j < len(completions2_turns)
                    and turn_idx < len(completions2_turns[j])
                    else ""
                )
                for j in range(len(test_cases))
            ]

            # Get metrics for this turn using the single-turn logger
            turn_metrics = code_reward_logger(
                [turn_completions1[i]],
                [turn_completions2[i]],
                [test_cases[i]],
                [entry_points[i]],
                [prompts[i]] if prompts else None,
            )

            if turn_metrics:
                turn_metric = turn_metrics[0]
                turn_rewards.append(turn_metric["total_reward"])

                # Add turn-specific metrics
                for key, value in turn_metric.items():
                    if key not in ["sample_id", "entry_point"]:
                        sample_metrics[f"turn_{turn_idx + 1}/{key}"] = value

                # Check for early termination
                if turn_metric["total_reward"] == 4.0:
                    sample_metrics["early_termination"] = True
                    sample_metrics["termination_turn"] = turn_idx + 1
                    sample_metrics["num_turns"] = turn_idx + 1
                    early_terminated = True
                    # Don't break - continue to fill remaining turns with max values

            sample_metrics["num_turns"] = turn_idx + 1

        # Calculate turn-to-turn improvements
        if len(turn_rewards) >= 2:
            for turn_idx in range(1, len(turn_rewards)):
                if (
                    f"turn_{turn_idx + 1}/improvement_from_turn_{turn_idx}"
                    not in sample_metrics
                ):
                    improvement = turn_rewards[turn_idx] - turn_rewards[turn_idx - 1]
                    sample_metrics[
                        f"turn_{turn_idx + 1}/improvement_from_turn_{turn_idx}"
                    ] = improvement

        # Overall metrics
        if turn_rewards:
            sample_metrics["overall/best_turn_reward"] = max(turn_rewards)
            sample_metrics["overall/final_turn_reward"] = turn_rewards[-1]
            sample_metrics["overall/avg_turn_reward"] = np.mean(turn_rewards)

        all_metrics.append(sample_metrics)

    return all_metrics


def aggregate_mt_humaneval_metrics_for_logging(
    metrics_list: List[Dict[str, Any]], num_turns: int = 2
) -> Dict[str, float]:
    """
    Aggregate multi-turn metrics from multiple samples for wandb logging.

    Args:
        metrics_list: List of sample metrics from mt_humaneval_logger
        num_turns: Number of turns to aggregate metrics for

    Returns:
        Dictionary of aggregated metrics
    """
    if not metrics_list:
        return {}

    aggregated = {}

    # Overall metrics
    overall_keys = ["num_turns", "early_termination"]
    for key in overall_keys:
        values = [sample[key] for sample in metrics_list if key in sample]
        if values:
            if key == "early_termination":
                aggregated[f"avg_{key}_rate"] = np.mean([1 if v else 0 for v in values])
            else:
                aggregated[f"avg_{key}"] = np.mean(values)

    # Turn-specific metrics
    for turn in range(1, num_turns + 1):
        turn_prefix = f"turn_{turn}"

        # Metrics to aggregate per turn
        turn_metrics = [
            "level_1_reward",
            "level_2_reward",
            "level_3_reward",
            "total_reward",
            "test_reward",
            "passed_tests",
            "total_tests",
            "passed_rate",
            "timeout_num",
            "bonus_reward",
            "aux_usage_bonus",
            "anti_wrapper_bonus",
            "called_wo_used_deduction",
            "gated_total_reward",
        ]

        for metric in turn_metrics:
            key = f"{turn_prefix}/{metric}"
            values = [sample[key] for sample in metrics_list if key in sample]
            if values:
                aggregated[f"{turn_prefix}/avg_{metric}"] = np.mean(values)

        # Improvement metrics
        if turn > 1:
            improvement_key = f"{turn_prefix}/improvement_from_turn_{turn - 1}"
            values = [
                sample[improvement_key]
                for sample in metrics_list
                if improvement_key in sample
            ]
            if values:
                aggregated[f"{turn_prefix}/avg_improvement"] = np.mean(values)

    # Overall aggregated metrics
    overall_reward_keys = [
        "overall/best_turn_reward",
        "overall/final_turn_reward",
        "overall/avg_turn_reward",
    ]
    for key in overall_reward_keys:
        values = [sample[key] for sample in metrics_list if key in sample]
        if values:
            aggregated[key.replace("overall/", "avg_")] = np.mean(values)

    return aggregated
