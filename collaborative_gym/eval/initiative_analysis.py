import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Dict

import dspy
import numpy as np
from knowledge_storm import TogetherClient, AzureOpenAIModel
from tqdm import tqdm

from collaborative_gym.core import SendTeammateMessage
from collaborative_gym.utils.utils import load_api_key


class JudgeInitiative(dspy.Signature):
    """I am analyzing team member initiative in collaboration. Two types of utterance count as taking initiative:
    1. Task Initiative: A team member is said to have the task initiative if she is directing how other member(s)' task should be accomplished, i.e., if her utterances directly propose actions that other members should perform.
        - Examples: “Let’s send engine E2 to Corning.”, “Let’s look at the first problem first.”, "Let's consider driving from Fort Lauderdale to Louisiana and explore three cities there."
        - Passive utterances like “Any suggestions”, "Right, okay." are not considered as task initiative.
    2. Dialogue Initiative: A team member is said to have the dialogue initiative if she tries to establish mutual beliefs. Both giving concrete information and asking concrete questions are considered dialogue initiative.
        - Examples: “We can’t go by Dansville because we’ve got Engine 1 going on that track.”, "Would you like to consider traveling on a different date?", "What do you think about the first problem?"
        - Repeating what the other person said, asking for clarification are not considered dialogue initiative.

    Now given an utterance in the conversation, you need to judge whether the utterance takes initiative or not. Indicate your judgement with "Yes" or "No".
    """

    utterance = dspy.InputField(prefix="Utterance: ", format=str)
    output = dspy.OutputField(
        prefix='Indicate your judgement with "Yes" or "No":', format=str
    )


class AnalyzeInitiative(dspy.Module):
    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.engine = lm
        self.judge_initiative = dspy.ChainOfThought(JudgeInitiative)

    def forward(self, utterances: List[str]):
        def judge_initiative(utterance: str):
            with dspy.settings.context(lm=self.engine, show_guidelines=False):
                initiative = self.judge_initiative(utterance=utterance).output
            return {
                "utterance": utterance,
                "initiative": initiative,
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for utterance in utterances:
                futures.append(executor.submit(judge_initiative, utterance))
            results = []
            for future in as_completed(futures):
                results.append(future.result())

        initiative_count = sum(
            [1 for result in results if "Yes" in result["initiative"]]
        )

        return dspy.Prediction(results=results, initiative_count=initiative_count)


class TeamInitiativeEvaluator:
    """
    Analyze team member initiative in collaboration.

    An utterance is considered to exhibit initiative if it directs task execution
    or facilitates mutual understanding within the collaboration. This class uses
    a language model to annotate utterances, with the `JudgeInitiative` method
    defining the prompt for evaluation.

    To quantify the distribution of initiative, we measure entropy:

        Initiative Entropy (H_init) =
        −Σ (p_i * logN(p_i)) for all i where p_i > 0,
        0 if any p_i = 0

    where N is the number of members in the team, and p_i represents the
    proportion of initiative-taking uttereances by team member i.
    """

    def __init__(self, lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.lm = lm
        self.send_message_action = SendTeammateMessage()
        self.analyze_initiative = AnalyzeInitiative(lm=self.lm)

    def gather_messages(self, event_log: List[Dict], role: str):
        messages = []

        for idx, event in enumerate(event_log):
            if (
                event["role"] == role
                and event["action_type"] == "collaborative"
                and event["action_status"] == "succeeded"
            ):
                action = event["action"]
                message = self.send_message_action.parse(action)["message"]
                messages.append(message)

        return messages

    def code_initiative(self, event_log):
        team_members = set([event["role"] for event in event_log])
        results = {}
        for team_member in team_members:
            messages = self.gather_messages(event_log, team_member)
            output = self.analyze_initiative(messages)
            results[team_member] = {
                "results": output.results,
                "initiative_count": output.initiative_count,
            }

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    load_api_key("secrets.toml")
    lm = AzureOpenAIModel(
        model="gpt-4o",
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_ENDPOINT"],
        api_version=os.environ["AZURE_API_VERSION"],
        max_tokens=50,
        temperature=0,
    )

    evaluator = TeamInitiativeEvaluator(lm=lm)

    results = {}
    for d in tqdm(os.listdir(args.result_dir)):
        if os.path.isdir(os.path.join(args.result_dir, d)):
            if os.path.exists(os.path.join(args.result_dir, d, "event_log.jsonl")):
                event_log = []
                with open(os.path.join(args.result_dir, d, "event_log.jsonl")) as f:
                    for line in f:
                        event_log.append(json.loads(line))
            else:
                print(f"Event log not found for {d}")
                continue
            results[d] = evaluator.code_initiative(event_log)

    # Save results
    with open(
        os.path.join(args.result_dir, "initiative_analysis_results.json"), "w"
    ) as f:
        json.dump(results, f, indent=4)

    # Compute aggregated results
    aggregated_results = {}
    initiative_entropy = []
    for session_id, result in results.items():
        all_initiative_count = 0
        for team_member, team_member_result in result.items():
            if (
                "user" in team_member
            ):  # FIXME: hard code for single-human-single-agent scenario
                team_member = "user"
            if team_member not in aggregated_results:
                aggregated_results[team_member] = {
                    "initiative_count": [],
                    "initiative_share": [],
                }
            aggregated_results[team_member]["initiative_count"].append(
                team_member_result["initiative_count"]
            )
            all_initiative_count += team_member_result["initiative_count"]

        entropy = 0
        for team_member in aggregated_results:
            if all_initiative_count == 0:
                aggregated_results[team_member]["initiative_share"].append(0)
            else:
                aggregated_results[team_member]["initiative_share"].append(
                    aggregated_results[team_member]["initiative_count"][-1]
                    / all_initiative_count
                )
                entropy -= aggregated_results[team_member]["initiative_share"][
                    -1
                ] * np.log2(
                    aggregated_results[team_member]["initiative_share"][-1]
                )  # FIXME: hard code as base 2 for single-human-single-agent scenario
        # handle NaN
        if np.isnan(entropy) or entropy < 0:
            entropy = 0
        initiative_entropy.append(entropy)

    for team_member, team_member_result in aggregated_results.items():
        print(f"Team member: {team_member}")
        print(
            f"Average initiative count: "
            f"{sum(team_member_result['initiative_count']) / len(team_member_result['initiative_count'])}"
        )
        print(
            f"Average initiative share: "
            f"{sum(team_member_result['initiative_share']) / len(team_member_result['initiative_share'])}"
        )
        print()

    # Initiative entropy
    print(
        f"Average initiative entropy: {sum(initiative_entropy) / len(initiative_entropy)}"
    )

    print("Token Usage:")
    print(lm.get_usage_and_reset())
