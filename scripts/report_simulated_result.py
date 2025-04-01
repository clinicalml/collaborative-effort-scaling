import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Union

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from task performance and event logs."
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory containing experiment session folders.",
    )
    return parser.parse_args()


def load_task_performance(session_dir):
    performance_path = os.path.join(session_dir, "task_performance.json")
    with open(performance_path) as f:
        return json.load(f)


def update_task_performance_if_needed(session_dir, task_performance):
    outcome = task_performance.get("outcome", "")
    if len(outcome.strip()) == 0:
        # Task not completed if outcome is empty.
        task_performance["task_completion"] = 0
        task_performance["performance_rating"] = 0
        performance_path = os.path.join(session_dir, "task_performance.json")
        with open(performance_path, "w") as f:
            json.dump(task_performance, f, indent=4)
    return task_performance


def process_event_log(session_dir):
    event_log_path = os.path.join(session_dir, "event_log.jsonl")
    events = []
    with open(event_log_path) as f:
        for line in f:
            events.append(json.loads(line))
    return events


def process_session(result_dir, session_name):
    session_dir = os.path.join(result_dir, session_name)
    task_performance = load_task_performance(session_dir)
    task_performance = update_task_performance_if_needed(session_dir, task_performance)

    total_case = 1
    complete = task_performance.get("task_completion", 0)
    performance_rating = task_performance.get("performance_rating", 0)
    collaboration_score = complete * performance_rating

    # Load event log and count actions
    events = process_event_log(session_dir)
    agent_action_cnt = user_action_cnt = 0
    agent_collab_cnt = user_collab_cnt = 0

    for event in events:
        role = event.get("role", "")
        if "agent" in role:
            agent_action_cnt += 1
            if event.get("action_type") == "collaborative":
                agent_collab_cnt += 1
        elif "user" in role:
            user_action_cnt += 1
            if event.get("action_type") == "collaborative":
                user_collab_cnt += 1

    return {
        "complete": complete,
        "performance_rating": performance_rating,
        "collaboration_score": collaboration_score,
        "agent_action_cnt": agent_action_cnt,
        "user_action_cnt": user_action_cnt,
        "agent_collab_cnt": agent_collab_cnt,
        "user_collab_cnt": user_collab_cnt,
    }, events


def aggregate_sessions(result_dir):
    total_sessions = 0
    complete_list = []
    performance_list = []
    collab_scores = []
    agent_actions = []
    user_actions = []
    agent_msgs = []
    user_msgs = []

    all_user_efforts = []
    session_names = []
    for session_name in os.listdir(result_dir):
        session_path = os.path.join(result_dir, session_name)
        if not os.path.isdir(session_path):
            continue

        if "_old" in session_name or ".del" in session_name:
            continue

        session_dir = os.path.join(result_dir, session_name)
        performance_path = os.path.join(session_dir, "task_performance.json")
        if not os.path.exists(performance_path):
            print(f"Task performance file not found for {session_name}")
            continue

        session_data, events = process_session(result_dir, session_name)
        total_sessions += 1
        complete_list.append(session_data["complete"])
        # if session_data["complete"] == 1:
        performance_list.append(session_data["performance_rating"])
        collab_scores.append(session_data["collaboration_score"])
        agent_actions.append(session_data["agent_action_cnt"])
        user_actions.append(session_data["user_action_cnt"])
        agent_msgs.append(session_data["agent_collab_cnt"])
        user_msgs.append(session_data["user_collab_cnt"])
        session_names.append(session_name)

        if "tabular_analysis" in session_name:
            user_effort_count_per_step = (
                analyze_context_length_for_session_tabular_analysis(events)
            )
        elif "travel_planning" in session_name:
            user_effort_count_per_step = (
                analyze_context_length_for_session_travel_planning(events)
            )
        else:
            print(f"Unknown session type: {session_name}")
            raise ValueError("Unknown session type")

        all_user_efforts.append(user_effort_count_per_step)

    if total_sessions == 0 or len(performance_list) == 0:
        print("No valid sessions found.")
        return

    aggregated_result = {
        "delivery_rate": sum(complete_list) / total_sessions,
        "task_performance": sum(performance_list) / len(performance_list),
        "collaboration_score": sum(collab_scores) / total_sessions,
        "average_agent_action_cnt": sum(agent_actions) / total_sessions,
        "avg_user_action_cnt": sum(user_actions) / total_sessions,
        "avg_agent_message_cnt": sum(agent_msgs) / total_sessions,
        "avg_user_message_cnt": sum(user_msgs) / total_sessions,
    }

    print(f"Delivery rate: {aggregated_result['delivery_rate']}")
    print(f"Task performance: {aggregated_result['task_performance']}")
    print(f"Collaboration score: {aggregated_result['collaboration_score']}")

    output_path = os.path.join(result_dir, "aggregated_result.json")
    with open(output_path, "w") as f:
        json.dump(aggregated_result, f, indent=4)

    all_results = {
        "session_names": session_names,
        "delivery_rate": complete_list,
        "task_performance": performance_list,
        "collaboration_score": collab_scores,
        "agent_action_cnt": agent_actions,
        "user_action_cnt": user_actions,
        "agent_message_cnt": agent_msgs,
        "user_message_cnt": user_msgs,
        "user_effort_count_per_step": all_user_efforts,
    }
    output_path = os.path.join(result_dir, "flatten_result.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)


def calculate_effort_for_edits(
    prev_code_history: str, current_code_history: str
) -> int:
    if prev_code_history == current_code_history:
        return 0
    if prev_code_history in current_code_history:
        return count_words(current_code_history[len(prev_code_history) :])

    return count_words(current_code_history)


def analyze_context_length_for_session_travel_planning(events: List[Dict]):
    df = pd.DataFrame(events)

    user_rows = df[df["role"].str.contains("user")]
    user_indices = user_rows.index

    user_effort_count_per_step = []

    for i in range(0, len(user_indices)):
        user_index = user_indices[i]
        current_row = df.iloc[user_index]
        # Two types of effort: (1) reading effort (2) executing effort

        # step 1: check the reading effort
        # We assume the effort comes from reading the agent's messages in the travel planning case

        if i == 0:
            prev_chat_history = df.iloc[0]["current_chat_history"]
            prev_travel_plan_editor = df.iloc[0]["current_observation"]["public"].get(
                "travel_plan_editor", ""
            )
        else:
            prev_chat_history = df.iloc[user_indices[i - 1]]["current_chat_history"]
            prev_travel_plan_editor = df.iloc[user_indices[i - 1]][
                "current_observation"
            ]["public"].get("travel_plan_editor", "")

        current_chat_history = current_row["current_chat_history"]

        new_chats = current_chat_history[len(prev_chat_history) :]
        new_chats_text = [
            ele["message"]
            for ele in new_chats
            if ele["role"] != "user" and ele["role"] != "environment"
        ]  # we only care about the new messages from the agent
        new_chats_word_count = sum(count_words(ele) for ele in new_chats_text)

        # Check the updates in the travel plan editor
        current_travel_plan_editor = current_row["current_observation"]["public"].get(
            "travel_plan_editor", ""
        )
        travel_plan_editor_effort = calculate_effort_for_edits(
            prev_travel_plan_editor, current_travel_plan_editor
        )

        # step 2: check the executing effort
        # There could be two types of executing effort: (1) the user chat or (2) user execution
        if current_row["action_type"] == "collaborative":
            # the user is just chatting via sending a message
            # extract the message from the action
            user_message = (
                current_row["action"]
                .replace("SEND_TEAMMATE_MESSAGE(message=", "")
                .rstrip(")")
            )
            user_message_word_count = count_words(user_message)
        elif current_row["action_type"] == "environment":
            # the user is executing an action
            # we assume the cost is mostly reading the outputs
            current_observation = current_row["current_observation"]
            output_word_count = count_words(
                current_observation["private"]["user"]
                .get("search_output", {})
                .get("output", "")
            )
            query_word_count = count_words(
                current_observation["private"]["user"]
                .get("search_output", {})
                .get("query", "")
            )
            user_action_word_count = output_word_count + query_word_count

        user_effort_count_per_step.append(
            {
                "global_step": int(user_indices[i]),
                "user_step": i,
                "chat_history_word_count": new_chats_word_count,
                "check_observation_word_count": travel_plan_editor_effort,
                "user_action_word_count": (
                    user_message_word_count
                    if current_row["action_type"] == "collaborative"
                    else user_action_word_count
                ),
                "user_action_type": current_row["action_type"],
            }
        )
    return user_effort_count_per_step


def analyze_context_length_for_session_tabular_analysis(events: List[Dict]):
    df = pd.DataFrame(events)

    user_rows = df[df["role"].str.contains("user")]
    user_indices = user_rows.index

    user_effort_count_per_step = []

    for i in range(0, len(user_indices)):
        user_index = user_indices[i]
        current_row = df.iloc[user_index]
        # Two types of effort: (1) reading effort (2) executing effort

        # step 1: check the reading effort
        # We assume the effort comes from reading the agent's messages in the travel planning case

        if i == 0:
            prev_chat_history = df.iloc[0]["current_chat_history"]
            prev_jupyter_history = df.iloc[0]["current_observation"]["public"].get(
                "jupyter_history", ""
            )
            prev_result_editor = df.iloc[0]["current_observation"]["public"].get(
                "result_editor", ""
            )
        else:
            prev_chat_history = df.iloc[user_indices[i - 1]]["current_chat_history"]
            prev_jupyter_history = df.iloc[user_indices[i - 1]]["current_observation"][
                "public"
            ].get("jupyter_history", "")
            prev_result_editor = df.iloc[user_indices[i - 1]]["current_observation"][
                "public"
            ].get("result_editor", "")

        current_chat_history = current_row["current_chat_history"]

        new_chats = current_chat_history[len(prev_chat_history) :]
        new_chats_text = [
            ele["message"]
            for ele in new_chats
            if ele["role"] != "user" and ele["role"] != "environment"
        ]  # we only care about the new messages from the agent
        new_chats_word_count = sum(count_words(ele) for ele in new_chats_text)

        # Check the updates in the observation compared to last step
        current_jupyter_history = current_row["current_observation"]["public"].get(
            "jupyter_history", ""
        )
        current_result_editor = current_row["current_observation"]["public"].get(
            "result_editor", ""
        )

        jupyter_effort = calculate_effort_for_edits(
            prev_jupyter_history, current_jupyter_history
        )
        result_editor_effort = calculate_effort_for_edits(
            prev_result_editor, current_result_editor
        )

        # step 2: check the executing effort
        # There could be two types of executing effort: (1) the user chat or (2) user execution
        if current_row["action_type"] == "collaborative":
            # the user is just chatting via sending a message
            # extract the message from the action
            user_message = (
                current_row["action"]
                .replace("SEND_TEAMMATE_MESSAGE(message=", "")
                .rstrip(")")
            )
            user_message_word_count = count_words(user_message)
        elif current_row["action_type"] == "environment":
            # the user is executing an action
            # we assume the cost is mostly reading the outputs
            current_observation = current_row["current_observation"]
            output_word_count = count_words(
                current_observation["private"]["user"]
                .get("search_output", {})
                .get("output", "")
            )
            query_word_count = count_words(
                current_observation["private"]["user"]
                .get("search_output", {})
                .get("query", "")
            )
            user_action_word_count = output_word_count + query_word_count

        user_effort_count_per_step.append(
            {
                "global_step": int(user_indices[i]),
                "user_step": i,
                "chat_history_word_count": new_chats_word_count,
                "check_observation_word_count": jupyter_effort + result_editor_effort,
                "user_action_word_count": (
                    user_message_word_count
                    if current_row["action_type"] == "collaborative"
                    else user_action_word_count
                ),
                "user_action_type": current_row["action_type"],
            }
        )
    return user_effort_count_per_step


def aggregate_sessions_effort(result_dir):

    for session_name in os.listdir(result_dir):
        session_path = os.path.join(result_dir, session_name)
        if not os.path.isdir(session_path):
            continue

        if "_old" in session_name or ".del" in session_name:
            continue

        session_data, events = process_event_log(session_path)


def main():
    args = parse_arguments()
    aggregate_sessions(args.result_dir)


if __name__ == "__main__":
    main()
