import glob
import json
from collections import defaultdict

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

# Download needed nltk resources (only needed once)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def count_words(text: str) -> int:
    """Count the number of words in a text using NLTK tokenizer."""
    if not text:
        return 0
    tokens = word_tokenize(text)
    return len(tokens)


def load_task_final_performance(base_dir: str, excluding_models: list = None):
    """
    Load only the final task performance data from task_performance.json files.

    Args:
        base_dir: Base directory containing the experiment data

    Returns:
        List of dictionaries containing task performance data with session_name and model_name
    """
    all_performance_data = []
    excluding_models = excluding_models or []

    for file_path in glob.glob(f"{base_dir}**/**/**/task_performance.json"):
        instance_name = file_path.split("/")[1]
        model_name = instance_name.split("_")[-1]

        if model_name in excluding_models:
            # print(f"Excluding model: {model_name}")
            continue

        session_name = file_path.split("/")[3]

        task_performance_file = file_path
        try:
            with open(task_performance_file, "r") as f:
                final_performance = json.load(f)

            # Add metadata to the performance data
            performance_record = {
                **final_performance,
                "session_name": session_name,
                "model_name": model_name,
            }

            all_performance_data.append(performance_record)

        except FileNotFoundError:
            print(f"Task performance file not found: {task_performance_file}")
            continue

    return all_performance_data


def load_agent_process_eval(
    base_dir: str,
    fill_empty_interactions: bool = False,
    pad_trajectories: bool = False,
    truncate_trajectories: bool = True,
):
    # base_dir = "paper-exp-simulated/travel_planning_one"

    max_steps = defaultdict(list)
    for file_path in glob.glob(f"{base_dir}**/**/agent_progress_eval.json"):
        instance_name = file_path.split("/")[1]
        model_name = instance_name.split("_")[-1]
        with open(file_path, "r") as f:
            progress_data = json.load(f)

        for key, eval_data in progress_data.items():
            eval_data = pd.DataFrame(eval_data)
            if len(eval_data) == 0:
                continue

            assert len(eval_data) == eval_data["user_step"].max() + 1
            max_steps[model_name].append(eval_data["user_step"].max() + 1)

    for key, value in max_steps.items():
        max_steps[key] = sorted(value)
        max_steps[key] = max_steps[key][
            int(len(max_steps[key]) * 0.05) : int(len(max_steps[key]) * 0.95)
        ]
        print(key, len(max_steps[key]), max_steps[key])

    all_eval_data = []
    for file_path in glob.glob(f"{base_dir}**/**/agent_progress_eval.json"):
        instance_name = file_path.split("/")[1]
        model_name = instance_name.split("_")[-1]

        with open(file_path, "r") as f:
            progress_data = json.load(f)

        for key, eval_data in progress_data.items():
            eval_data = pd.DataFrame(eval_data)

            if len(eval_data) == 0:
                task_performance_eval_file = file_path.replace(
                    "agent_progress_eval.json", f"{key}/task_performance.json"
                )
                if not fill_empty_interactions:
                    print(task_performance_eval_file)
                    continue
                else:
                    with open(task_performance_eval_file, "r") as f:
                        final_performance = json.load(f)
                    # print(final_performance)
                    eval_data = pd.DataFrame(
                        [
                            {
                                "global_step": 0,
                                "user_step": 0,
                                "has_result_updates": True,
                                "last_result_update_content": None,
                                "evaluation": final_performance,
                                "performance_rating": final_performance[
                                    "performance_rating"
                                ],
                                # "commonsense_pass_rate": final_performance.get(
                                #     "commonsense_pass_rate", 0
                                # ),
                                # "preference_pass_rate": final_performance.get(
                                #     "preference_pass_rate", 0
                                # ),
                            }
                        ]
                    )
            else:

                # eval_data["user_action_tokens"] = eval_data["user_action"].apply(
                #     count_words
                # )
                # eval_data["agent_action_tokens"] = eval_data["agent_actions"].apply(
                #     lambda row: (
                #         sum([count_words(ele) for ele in filter_collaborative_actions(row)])
                #         if isinstance(row, list)
                #         else 0
                #     )
                # )

                eval_data["performance_rating"] = (
                    eval_data["performance_rating"].fillna(method="ffill").fillna(0)
                )

                # eval_data["commonsense_pass_rate"] = eval_data["evaluation"].apply(
                #     lambda x: (
                #         x.get("commonsense_pass_rate", None)
                #         if isinstance(x, dict)
                #         else None
                #     )
                # )
                # eval_data["preference_pass_rate"] = eval_data["evaluation"].apply(
                #     lambda x: (
                #         x.get("preference_pass_rate", None)
                #         if isinstance(x, dict)
                #         else None
                #     )
                # )

                # eval_data["commonsense_pass_rate"] = (
                #     eval_data["commonsense_pass_rate"].fillna(method="ffill").fillna(0)
                # )
                # eval_data["preference_pass_rate"] = (
                #     eval_data["preference_pass_rate"].fillna(method="ffill").fillna(0)
                # )

            # effort_factor = 0.25
            # eval_data["combined_token_count"] = (
            #     eval_data["user_action_tokens"]
            #     + eval_data["agent_action_tokens"] * effort_factor
            # )
            # eval_data["accumulated_effort"] = eval_data["combined_token_count"].cumsum()

            max_step = max(max_steps[model_name])
            # Fill to the max step
            if len(eval_data) < max_step and pad_trajectories:
                eval_data = pd.concat(
                    [
                        eval_data,
                        pd.DataFrame(
                            [
                                {
                                    "global_step": None,
                                    "user_step": i,
                                    "has_result_updates": False,
                                    "last_result_update_content": None,
                                    "evaluation": {},
                                    "performance_rating": eval_data.performance_rating.iloc[
                                        -1
                                    ],
                                }
                                for i in range(len(eval_data), max_step)
                            ]
                        ),
                    ]
                )
            if truncate_trajectories:
                eval_data = eval_data.iloc[:max_step]

            eval_data["session_name"] = key
            eval_data["model_name"] = model_name

            all_eval_data.append(eval_data)

    return all_eval_data


def load_agent_process_likert_score(
    base_dir: str,
    fill_empty_interactions: bool = False,
    pad_trajectories: bool = False,
    truncate_trajectories: bool = True,
):
    all_eval_data = []
    for file_path in glob.glob(f"{base_dir}**/**/likert_score.json"):
        instance_name = file_path.split("/")[1]
        model_name = instance_name.split("_")[-1]

        with open(file_path, "r") as f:
            progress_data = json.load(f)

        for key, eval_data in progress_data.items():
            eval_data = pd.DataFrame(eval_data)

            eval_data["session_name"] = key
            eval_data["model_name"] = model_name
            all_eval_data.append(eval_data)

    return all_eval_data
