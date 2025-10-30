import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import dspy
import nltk
import pandas as pd
from knowledge_storm import AzureOpenAIModel
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from collaborative_gym.core import SendTeammateMessage
from collaborative_gym.envs.literature_survey import *
from collaborative_gym.envs.tabular_analysis import *
from collaborative_gym.envs.travel_planning import *
from collaborative_gym.utils.utils import load_api_key

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


class JudgeAgentProgress(dspy.Signature):
    """Given a previous user message, agent's response/question, and the next user message,
    judge whether the agent is making progress in addressing the user's needs.
    Provide a rating on a 5-point Likert scale:
    1: Strongly Disagree - Agent made no progress or moved backwards
    2: Disagree - Agent made minimal progress
    3: Neutral - Agent maintained the same level of progress
    4: Agree - Agent made good progress
    5: Strongly Agree - Agent made excellent progress
    Output the rating (1-5)."""

    previous_user_message = dspy.InputField(
        prefix="Previous User Message: ", format=str
    )
    agent_message = dspy.InputField(prefix="Agent Message: ", format=str)
    next_user_message = dspy.InputField(prefix="Next User Message: ", format=str)
    rating = dspy.OutputField(
        prefix="Indicate your rating with a single number among 1/2/3/4/5, and if you want to provide an explanation, please put it after a new line: ",
        format=str,
    )


class AnalyzeAgentProgress:

    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel], task_name: str):
        super().__init__()
        self.engine = lm
        self.task_name = task_name
        self.judge_agent_progress = dspy.ChainOfThought(JudgeAgentProgress)

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
                launch_jupyter=False,
            )
            editor_update_action = eval_env.action_space[1]
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")
        return eval_env, editor_update_action

    def obtain_agent_collaborative_actions(self, agent_actions):
        if self.task_name == "travel_planning":
            action_types = ["EDITOR_UPDATE", "SEND_TEAMMATE_MESSAGE"]
            subset = agent_actions["action"].apply(
                lambda action: any(
                    action_type in action for action_type in action_types
                )
            )
            agent_actions = agent_actions[subset].copy()
        elif self.task_name == "tabular_analysis":
            action_types = [
                "EDITOR_UPDATE",
                "SEND_TEAMMATE_MESSAGE",
                "EXECUTE_JUPYTER_CELL",
            ]
            subset = agent_actions["action"].apply(
                lambda action: any(
                    action_type in action for action_type in action_types
                )
            )
            agent_actions = agent_actions[subset].copy()
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")
        return agent_actions

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
                user_action = df.loc[0]["action"]
            elif i == len(user_indices):
                user_action = df.loc[user_indices[i - 1]]["action"]
                agent_actions = df.loc[user_indices[i - 1] :]
            else:
                agent_actions = df.loc[user_indices[i - 1] + 1 : user_indices[i] - 1]
                user_action = df.loc[user_indices[i - 1]]["action"]

            # display(agent_actions)
            # display(df.loc[user_indices[i:i+1]])

            editor_update_rows = agent_actions["action"][
                agent_actions["action"].str.contains("EDITOR_UPDATE")
            ].tolist()
            has_result_updates = len(editor_update_rows) > 0

            interaction_session = {
                "global_step": int(user_indices[i - 1]) if i > 0 else 0,
                "user_step": i,
                "user_action": user_action,
                # "user_action_tokens": count_words(user_action),
                "agent_actions": agent_actions["action"].to_list(),
                # "agent_actions_tokens": sum([
                #     count_words(action) for action in agent_actions["action"].to_list()
                # ]),
                "has_result_updates": has_result_updates,
                "last_result_update_content": (
                    editor_update_rows[-1] if has_result_updates else None
                ),
                "rating": None,
                "explanation": None,
            }

            return interaction_session

        if len(user_indices) == 0:
            return []

        output = [process_user_interaction(i) for i in range(len(user_indices) + 1)]
        combined_df = pd.DataFrame(output)
        combined_df["next_user_action"] = combined_df["user_action"].shift(-1)

        def score_step(idx):

            row_data = combined_df.iloc[idx].to_dict()
            agent_actions = row_data["agent_actions"]
            agent_message = "\n".join(
                [f"{idx+1}. {msg}" for idx, msg in enumerate(agent_actions)]
            )

            previous_user_action = row_data["user_action"]
            next_user_action = row_data["next_user_action"]

            if len(agent_actions) == 0:
                return row_data
            else:
                with dspy.settings.context(lm=self.engine, show_guidelines=False):
                    progress_result = self.judge_agent_progress(
                        previous_user_message=previous_user_action,
                        agent_message=agent_message,
                        next_user_message=next_user_action,
                    )

                try:
                    rating = "".join(
                        re.findall(r"\d", progress_result.rating.split("\n")[0])
                    )
                except (ValueError, AttributeError):
                    rating = None

                row_data["rating"] = rating
                row_data["explanation"] = progress_result.rating
                return row_data

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(len(user_indices)):
                futures.append(executor.submit(score_step, i))

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        progress_assessment_results = sorted(results, key=lambda x: x["global_step"])

        del eval_env
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
    lm = AzureOpenAIModel(
        model="gpt-4o",
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_ENDPOINT"],
        api_version=os.environ["AZURE_API_VERSION"],
        max_tokens=50,  # Increased to accommodate explanation
        temperature=0,
    )

    evaluator = AnalyzeAgentProgress(lm=lm, task_name=args.task)

    results = {}
    all_dirs = [
        d
        for d in os.listdir(args.result_dir)
        if not (".json" in d or "_old" in d or ".del" in d)
        and os.path.isdir(os.path.join(args.result_dir, d))
    ]

    all_dirs = sorted(all_dirs, key=lambda x: int(x.split("_")[-1]))

    if args.debug:
        all_dirs = all_dirs[:2]

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
            elif os.path.exists(os.path.join(args.result_dir, d, "event_log.json")):
                with open(os.path.join(args.result_dir, d, "event_log.json")) as f:
                    event_log = json.load(f)
            else:
                print(f"Event log not found for {d}")
                continue

            example_id = int(d.split("_")[-1])

            prediction = evaluator.forward(event_log, example_id=example_id)
            results[d] = prediction

    # Save results
    with open(os.path.join(args.result_dir, "likert_score.json"), "w") as f:
        json.dump(results, f, indent=4)
