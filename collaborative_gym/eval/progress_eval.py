import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import dspy
import pandas as pd
from knowledge_storm import AzureOpenAIModel
from tqdm import tqdm

from collaborative_gym.core import SendTeammateMessage
from collaborative_gym.envs.literature_survey import *
from collaborative_gym.envs.tabular_analysis import *
from collaborative_gym.envs.travel_planning import *
from collaborative_gym.utils.utils import load_api_key


class AnalyzeAgentProgress:

    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel], task_name: str):
        super().__init__()
        self.engine = lm
        self.task_name = task_name

    @staticmethod
    def is_agent_message(event):
        return (
            "agent" in event["role"]
            and event["action_type"] == "collaborative"
            and event["action_status"] == "succeeded"
        )

    @staticmethod
    def is_human_message(event):
        return (
            "user" in event["role"]
            and event["action_type"] == "collaborative"
            and event["action_status"] == "succeeded"
        )

    def load_environment(self, idx):
        if self.task_name == "travel_planning":
            eval_env = CoTravelPlanningEnv(
                team_members=[],
                env_id=f"eval_travel_planning_{idx}",
                use_simulated_dataset=True,
                travel_planner_data_point_idx=idx,
            )
            editor_update_action = eval_env.action_space[0]
        elif self.task_name == "tabular_analysis":
            eval_env = CoAnalysisEnv(
                team_members=[],
                env_id=f"eval_tabular_analysis_{idx}",
                use_simulated_dataset=True,
                discovery_bench_data_point_idx=idx,
            )
            editor_update_action = eval_env.action_space[1]
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")
        return eval_env, editor_update_action

    def forward(self, event_log, example_id):
        # Extract conversation history with timestamps

        eval_env, editor_update_action = self.load_environment(example_id)

        df = pd.DataFrame(event_log)
        user_indices = df[
            df["role"].str.contains("user")
            & df["action_type"].str.contains("collaborative")
        ].index

        def process_user_interaction(i):
            if i == 0:
                agent_actions = df.loc[1 : user_indices[i] - 1]
            else:
                agent_actions = df.loc[user_indices[i - 1] + 1 : user_indices[i] - 1]

            # display(agent_actions)
            # display(df.loc[user_indices[i:i+1]])

            editor_update_rows = agent_actions["action"][
                agent_actions["action"].str.contains("EDITOR_UPDATE")
            ].tolist()
            has_result_updates = len(editor_update_rows) > 0

            interaction_session = {
                "global_step": int(user_indices[i]),
                "user_step": i,
                "agent_actions": agent_actions["action"].to_list(),
                "user_action": df.loc[user_indices[i]]["action"],
                "has_result_updates": has_result_updates,
                "last_result_update_content": (
                    editor_update_action.parse(editor_update_rows[-1])
                    if has_result_updates
                    else None
                ),
                "evaluation": None,
                "performance_rating": None,
            }

            if has_result_updates:

                eval_output = eval_env._evaluate_task_performance(
                    interaction_session["last_result_update_content"],
                )
                interaction_session["evaluation"] = eval_output
                interaction_session["performance_rating"] = eval_output[
                    "performance_rating"
                ]

            return interaction_session

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(len(user_indices)):
                futures.append(executor.submit(process_user_interaction, i))

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        progress_assessment_results = sorted(results, key=lambda x: x["global_step"])
        return progress_assessment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--debug", action="store_true")  # only use a few examples
    parser.add_argument(
        "--skip_existing", action="store_true"
    )  # only use a few examples

    args = parser.parse_args()

    load_api_key("secrets.toml")
    lm = None

    evaluator = AnalyzeAgentProgress(lm=lm, task_name=args.task)

    results = {}
    all_dirs = [
        d
        for d in os.listdir(args.result_dir)
        if not ("_old" in d or ".del" in d or ".json" in d)
    ]
    all_dirs = sorted(all_dirs, key=lambda x: int(x.split("_")[-1]))

    if args.debug:
        all_dirs = all_dirs[:10]

    for d in tqdm(all_dirs):
        if os.path.isdir(os.path.join(args.result_dir, d)):

            if args.skip_existing and os.path.exists(
                os.path.join(args.result_dir, d, "agent_progress_eval.json")
            ):
                print(f"Skipping {d} as it already exists")
                continue

            if os.path.exists(os.path.join(args.result_dir, d, "event_log.jsonl")):
                event_log = []
                with open(os.path.join(args.result_dir, d, "event_log.jsonl")) as f:
                    for line in f:
                        event_log.append(json.loads(line))
            else:
                print(f"Event log not found for {d}")
                continue

            example_id = int(d.split("_")[-1])

            prediction = evaluator.forward(event_log, example_id=example_id)
            results[d] = prediction

    # Save results
    with open(os.path.join(args.result_dir, "agent_progress_eval.json"), "w") as f:
        json.dump(results, f, indent=4)
