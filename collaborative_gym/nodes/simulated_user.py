import asyncio
import json
import os
import random
import re
from typing import AsyncIterator, Dict, Union

import dspy
from aact import Message, NodeFactory
from knowledge_storm import AzureOpenAIModel

from collaborative_gym import JsonObj
from collaborative_gym.core import SendTeammateMessage, WaitTeammateContinue, logger
from collaborative_gym.nodes.base_node import BaseNode
from collaborative_gym.utils.context_processing import ContextProcessor
from collaborative_gym.utils.utils import serialize_llm_response


class UserPersona:
    """
    Defines and manages simulated user personality traits.

    Handles user behavior dimensions like proactiveness in feedback and actions.
    Used to create consistent but configurable user simulation behaviors.

    Attributes:
        dimensions: Dictionary mapping behavior dimensions to possible values
            - proactive_feedback: Whether user actively provides feedback
            - proactive_action: Whether user takes initiative in actions
    """

    def __init__(self):
        self.dimensions = {
            "proactive_feedback": [True, False],
            "proactive_action": [True, False],
        }

    def sample(self):
        """
        Generate a random user persona.

        Randomly selects values for each behavior dimension to create
        a complete personality profile.

        Returns:
            dict: Mapping of behavior dimensions to selected values
        """
        persona = {}
        for dim in self.dimensions:
            persona[dim] = random.choice(self.dimensions[dim])
        return persona


class DecideAction(dspy.Signature):
    """You are a user interacting with an agent to complete a task.
    Based on the current observation and chat history, decide what action to take next by choosing one of the followings.
    1. Answer the question: Choose this action if there is a question in the chat history waiting for your response.
    2. Offer feedback: Choose this action if the current observation is incorrect or deviates from the additional information you know.
    3. Take a task action: Choose this action if you want to take an action to help complete the task.
    4. Finish the task: Choose this action if you are satisfied with the current status of the task and want to finish it.
    5. Do nothing: Choose this action if there is no major issue and you want the agent to proceed.
    """

    rules = dspy.InputField(prefix="Rules for selecting your action:\n", format=str)
    task_description = dspy.InputField(
        prefix="The task description you initially sent to the agent:\n", format=str
    )
    observation = dspy.InputField(
        prefix="Current observation that reveals the current status of the task environment:\n",
        format=str,
    )
    chat_history = dspy.InputField(
        prefix="Current chat history between you and other teammates (e.g., the agent):\n",
        format=str,
    )
    available_actions = dspy.InputField(
        prefix='Available task actions you can take if you choose "3. Take a task action":\n',
        format=str,
    )
    additional_info = dspy.InputField(
        prefix="Additional information that you know (you can use the information to help the agent better complete your request):\n",
        format=str,
    )
    action_history = dspy.InputField(
        prefix="Actions you have already take (don't repeat the same action): ",
        format=str,
    )
    output = dspy.OutputField(
        prefix="The action you want to take next (Please output 1/2/3/4/5): ",
        format=str,
    )


class AnswerQuestion(dspy.Signature):
    """You are a user interacting with an agent to complete a task. Answer the question in the chat history based on the additional information you know.
    Rules:
    - You will stick to or fully utilize the additional information that only you know.
    - Just generate one line for the message to simulate a real user's behavior. Try to make the message as natural as possible.
    - Do not give away all the additional information at once. Only provide the information that is necessary for the question. You are a lazy user so you only provide one piece of information at a time.
    - Do not hallucinate information that is not provided in the additional information. For example, if the agent asks for something but it is not mentioned in the given information, do not make it up, just say you do not remember or have it.
    - Do not repeat the exact additional information in the answer. Instead, use your own words to convey the same information.
    """

    task_description = dspy.InputField(
        prefix="The task you want the agent to assist with:\n", format=str
    )
    observation = dspy.InputField(
        prefix="Current observation that reveals the current status of the task environment:\n",
        format=str,
    )
    chat_history = dspy.InputField(
        prefix="Current chat history between you and other teammates (e.g., the agent):\n",
        format=str,
    )
    additional_info = dspy.InputField(
        prefix="Additional information that only you know:\n", format=str
    )
    output = dspy.OutputField(
        prefix="The answer to the question in the chat history: ", format=str
    )


class OfferFeedback(dspy.Signature):
    """You are a user interacting with an agent to complete a task. Offer feedback to the agent based on the current observation and additional information you know.
    Rules:
    - You will stick to or fully utilize the additional information that only you know.
    - Just generate one line for the message to simulate a real user's behavior. Try to make the feedback as natural as possible.
    - Do not give away all the additional information at once. Be specific about what the agent did wrong or what information is missing.
    - Do not hallucinate feedback that is not based on the current observation or the additional information you know. If you have to answer, just say you do not know.
    - Do not repeat the exact additional information in the feedback. Instead, use your own words to convey the same information.
    """

    task_description = dspy.InputField(
        prefix="The task you want the agent to assist with:\n", format=str
    )
    observation = dspy.InputField(
        prefix="Current observation that reveals the current status of the task environment:\n",
        format=str,
    )
    chat_history = dspy.InputField(
        prefix="Current chat history between you and other teammates (e.g., the agent):\n",
        format=str,
    )
    additional_info = dspy.InputField(
        prefix="Additional information that only you know:\n", format=str
    )
    output = dspy.OutputField(
        prefix="The feedback you want to provide to the agent: ", format=str
    )


class TakeTaskAction(dspy.Signature):
    """You are a user interacting with an agent to complete a task. Take a task action to help complete the task.
    Note that you will stick to or fully utilize the additional information that only you know to help you take the action.
    """

    task_description = dspy.InputField(
        prefix="The task you want the agent to assist with:\n", format=str
    )
    observation = dspy.InputField(
        prefix="Current observation that reveals the current status of the task environment:\n",
        format=str,
    )
    chat_history = dspy.InputField(
        prefix="Current chat history between you and other teammates (e.g., the agent):\n",
        format=str,
    )
    action_space_description = dspy.InputField(
        prefix="You can choose from and only from the following actions. Note that these actions are only for "
        "interacting with the environment and cannot be executed as real code. Please strictly follow the "
        "action space specification. You can only choose one action at a time. Invalid actions will hurt your "
        "performance rating. The following actions are available:\n",
        format=str,
    )
    additional_info = dspy.InputField(
        prefix="Additional information that only you know:\n", format=str
    )
    output = dspy.OutputField(
        prefix="Action (the action string must follow the regex pattern of the selected action so it can be parsed later): ",
        format=str,
    )


class GetActionModule(dspy.Module):
    """
    DSPy module for determining and executing simulated user actions.

    Manages the decision-making process for simulated user actions using two language
    models: one for planning (deciding what type of action to take) and another for
    executing (generating the specific action content).

    Type Parameters:
        planning_lm: Language model for high-level action planning
        executing_lm: Language model for detailed action generation

    Attributes:
        planning_lm: LM instance for action planning
        executing_lm: LM instance for action execution
        decide_action: Chain-of-thought module for action selection
        answer_question: Chain-of-thought module for question responses
        offer_feedback: Chain-of-thought module for feedback generation
        take_task_action: Chain-of-thought module for task actions
        send_teammate_message_action: Action template for messages
        wait_teammate_continue_action: Action template for wait states
    """

    def __init__(
        self,
        planning_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        executing_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.planning_lm = planning_lm
        self.executing_lm = executing_lm
        self.decide_action = dspy.ChainOfThought(DecideAction)
        self.answer_question = dspy.ChainOfThought(AnswerQuestion)
        self.offer_feedback = dspy.ChainOfThought(OfferFeedback)
        self.take_task_action = dspy.ChainOfThought(TakeTaskAction)
        self.send_teammate_message_action = SendTeammateMessage()
        self.wait_teammate_continue_action = WaitTeammateContinue()

    def forward(
        self,
        persona: dict,
        task_description: str,
        observation: str,
        chat_history: str,
        action_history: str,
        task_action_names: str,
        task_action_space_description: str,
        private_information: str,
    ):
        rules = """- You will use your additional information to supervise the agent to help it better complete your request.
- You will ALWAYS choose "1. Answer the question" if there is a question in the chat history waiting for your response. Even if you don't have the answer, you will still choose this option to say you don't know.
- You care about task quality. Choose "4. Finish the task" only when you think your request is satisfied and the current task outcome is good.
- You also care about your time. Choose "4. Finish the task" if the task outcome is not empty and you have sent multiple repeated messages to the agent.
- You don't want to debug code for the agent. Just let the agent figure it out when there is any error in its code.
- Be patient. If the agent hasn't responded to your previous message, don't rush it. Give it some time to process the information."""
        if not persona["proactive_feedback"]:
            rules += '\n- You are not proactive in giving feedback so you will NEVER choose "2. Offer feedback". But you will still answer agent\'s questions (if any) by choosing "1. Answer the question".'
        else:
            rules += "\n- You are a lazy user but you care about the quality of the agent's work. You will only choose \"2. Offer feedback\" if you think the agent's actions deviate from the additional information you know."
        if not persona["proactive_action"]:
            rules += '\n- You are not proactive in taking actions so you will NEVER choose "3. Take a task action".'
        else:
            rules += "\n- You are a lazy user hoping the AI agent can do most of the job. You don't choose \"3. Take a task action\" unless you need to edit the task outcome. You don't want to take other task actions and prioritize waiting for the agent to continue the task."

        # Hacky patch
        chat_history = chat_history.replace(
            "The user can also send a message.", "You can also send a message."
        )

        # Decide what action to take next
        with dspy.settings.context(lm=self.planning_lm, show_guidelines=False):
            plan = self.decide_action(
                rules=rules,
                task_description=f"```\n{task_description}\n```",
                observation=observation,
                chat_history=chat_history,
                available_actions=task_action_names,
                action_history=action_history,
                additional_info=private_information,
            ).output
        if "1" in plan:
            # Answer the question
            logger.info("Simulated User: Answering the question")
            with dspy.settings.context(lm=self.executing_lm, show_guidelines=False):
                question = self.answer_question(
                    task_description=task_description,
                    observation=observation,
                    chat_history=chat_history,
                    additional_info=private_information,
                ).output
            action_str = (
                self.send_teammate_message_action.construct_action_string_from_params(
                    message=question
                )
            )
        elif "2" in plan and persona["proactive_feedback"]:
            # Offer feedback
            logger.info("Simulated User: Offering feedback")
            with dspy.settings.context(lm=self.executing_lm, show_guidelines=False):
                feedback = self.offer_feedback(
                    task_description=task_description,
                    observation=observation,
                    chat_history=chat_history,
                    additional_info=private_information,
                ).output
            action_str = (
                self.send_teammate_message_action.construct_action_string_from_params(
                    message=feedback
                )
            )
        elif "3" in plan and persona["proactive_action"]:
            # Take a task action
            logger.info("Simulated User: Taking a task action")
            with dspy.settings.context(lm=self.executing_lm, show_guidelines=False):
                action = self.take_task_action(
                    task_description=task_description,
                    observation=observation,
                    chat_history=chat_history,
                    action_space_description=task_action_space_description,
                    additional_info=private_information,
                ).output
            # Hacky post-processing:
            # Assume the action is in a function call format and the function name starts with a capital letter.
            match = re.search(r"[A-Z]", action)
            if match:
                action = action[match.start() :]
            if action[-1] != ")":
                action = action[: action.rfind(")") + 1]
            action = action.replace("\(", "(").replace("\)", ")")
            action_str = action
        elif "4" in plan:
            # Finish the task
            logger.info("Simulated User: Finishing the task")
            action_str = "FINISH()"
        else:
            # Do nothing
            logger.info(f"Simulated User: Doing nothing (raw plan: {plan})")
            action_str = (
                self.wait_teammate_continue_action.construct_action_string_from_params()
            )

        return dspy.Prediction(plan=plan, action_str=action_str)


class SimulatedUserProxy:
    """
    High-level controller for LLM-based user simulation.

    Manages the complete lifecycle of a simulated user, including initialization,
    action generation, and result logging. Uses language models to generate human-like
    responses based on task context (including hidden information not visible to the agent)
    and persona settings.

    When a new observation is received, the simulated user uses LM to decide the next action type:
    - Answer Question
    - Provide Feedback
    - Take Task Action
    - Do Nothing
    - Finish Task
    Based on the decision, the simulated user generates the specific action content using the LM.
    The action string is sent to the environment through the Redis channel.
    """

    def __init__(
        self,
        planning_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        executing_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        randomize_persona: bool = False,
        proactive_feedback: bool = True,
        proactive_action: bool = True,
    ):
        self.name = None
        self.team_members = None
        self.task_description = None
        self.task_action_names = None
        self.task_action_space_description = None
        self.planning_lm = planning_lm
        self.executing_lm = executing_lm
        self.private_information = None
        self.collaboration_acts = {
            "send_teammate_message": SendTeammateMessage(),
            "wait_teammate_continue": WaitTeammateContinue(),
        }
        self.context_processor = ContextProcessor()
        self.get_action_module = GetActionModule(
            planning_lm=self.planning_lm, executing_lm=self.executing_lm
        )
        self.action_history = []

        self.persona = None
        self.randomize_persona = randomize_persona
        self.proactive_feedback = proactive_feedback
        self.proactive_action = proactive_action

    def start(
        self,
        name: str,
        team_members: list[str],
        task_description: str,
        action_space: dict,
        private_information: dict,
    ):
        self.name = name
        self.team_members = team_members
        self.task_description = task_description
        self.task_action_space_description = self.context_processor.action_space_to_str(
            action_space=action_space,
            excluded_action_names=[
                self.collaboration_acts[k].human_readable_name
                for k in self.collaboration_acts
            ],
        )
        self.task_action_names = [act["human_readable_name"] for act in action_space]
        for k in self.collaboration_acts:
            if self.collaboration_acts[k].human_readable_name in self.task_action_names:
                self.task_action_names.remove(
                    self.collaboration_acts[k].human_readable_name
                )
        self.private_information = private_information
        if self.randomize_persona:
            self.persona = UserPersona().sample()
        else:
            self.persona = {
                "proactive_feedback": self.proactive_feedback,
                "proactive_action": self.proactive_action,
            }

    def get_action(self, observation: Dict, chat_history: list):
        observation_str = self.context_processor.observation_to_str(obs=observation)
        chat_history_str = self.context_processor.chat_history_to_str(
            current_role=self.name, chat_history=chat_history
        )
        output = self.get_action_module.forward(
            persona=self.persona,
            task_description=self.task_description,
            observation=observation_str,
            chat_history=chat_history_str,
            action_history="\n".join([act["action"] for act in self.action_history]),
            task_action_names=", ".join(self.task_action_names),
            task_action_space_description=self.task_action_space_description,
            private_information=json.dumps(self.private_information, indent=4),
        )

        self.action_history.append({"plan": output.plan, "action": output.action_str})

        return output.action_str

    def end(self, result_dir: str):
        os.makedirs(os.path.join(result_dir, "simulated_user"), exist_ok=True)
        with open(os.path.join(result_dir, "simulated_user", "info.json"), "w") as f:
            info = {
                "planning_lm": self.planning_lm.kwargs["model"],
                "executing_lm": self.executing_lm.kwargs["model"],
                "token_usage": self.get_token_usage(),
                "persona": self.persona,
            }
            json.dump(info, f, indent=4)
        with open(
            os.path.join(result_dir, "simulated_user", "llm_call_history.jsonl"), "w"
        ) as f:
            for call in self.planning_lm.history:
                f.write(
                    json.dumps({"prompt": call["prompt"], "response": serialize_llm_response(call["response"])})
                    + "\n"
                )
            for call in self.executing_lm.history:
                f.write(json.dumps(serialize_llm_response(call)) + "\n")
        with open(
            os.path.join(result_dir, "simulated_user", "action_history.json"), "w"
        ) as f:
            json.dump(self.action_history, f, indent=4)

    def get_token_usage(self):
        return {
            "planning_lm": {
                "prompt_tokens": self.planning_lm.prompt_tokens,
                "completion_tokens": self.planning_lm.completion_tokens,
            },
            "executing_lm": {
                "prompt_tokens": self.executing_lm.prompt_tokens,
                "completion_tokens": self.executing_lm.completion_tokens,
            },
        }


@NodeFactory.register("simulated_user")
class SimulatedUserNode(BaseNode[JsonObj, JsonObj]):
    """
    Asynchronous node for managing simulated user interactions in collaborative environments.

    Handles the communication between the simulated user and the environment through
    Redis channels. Manages message processing, action generation, and state tracking
    with support for concurrent operations.

    Type Parameters:
        JsonObj: Both input and output message types use JSON-serializable objects

    Attributes:
        env_uuid: Unique identifier for the environment instance
        node_name: Name/role of this simulated user
        simulated_user: Instance of SimulatedUserProxy handling behavior
        tasks: List of active async tasks
        is_processing_observation: Flag to prevent concurrent observation processing
        is_processing_observation_lock: AsyncIO lock for observation handling
    """

    def __init__(
        self,
        env_uuid: str,
        node_name: str,
        randomize_persona: bool = False,
        proactive_feedback: bool = True,
        proactive_action: bool = True,
        redis_url: str = "redis://localhost:6379/0",
    ):
        super().__init__(
            input_channel_types=[
                (f"{env_uuid}/{node_name}/observation", JsonObj),
                (f"{env_uuid}/start", JsonObj),
                (f"{env_uuid}/end", JsonObj),
            ],
            output_channel_types=[(f"{env_uuid}/step", JsonObj)],
            redis_url=redis_url,
        )
        self.env_uuid = env_uuid
        self.node_name = node_name
        self.simulated_user = SimulatedUserProxy(
            planning_lm=AzureOpenAIModel(
                model="gpt-4o",
                api_key=os.environ["AZURE_API_KEY"],
                azure_endpoint=os.environ["AZURE_ENDPOINT"],
                api_version=os.environ["AZURE_API_VERSION"],
                max_tokens=100,
            ),
            executing_lm=AzureOpenAIModel(
                model="gpt-4o",
                api_key=os.environ["AZURE_API_KEY"],
                azure_endpoint=os.environ["AZURE_ENDPOINT"],
                api_version=os.environ["AZURE_API_VERSION"],
                max_tokens=4000,
            ),
            randomize_persona=randomize_persona,
            proactive_feedback=proactive_feedback,
            proactive_action=proactive_action,
        )

        self.tasks = []
        self.is_processing_observation = False
        self.is_processing_observation_lock = asyncio.Lock()

    async def event_loop(self) -> None:
        """
        Main event processing loop for handling simulated user interactions.

        Manages concurrent task processing and ensures proper handling of
        observations with locking mechanisms to prevent race conditions.

        Returns:
            None
        """
        self.tasks = []
        async for input_channel, input_message in self._wait_for_input():
            if input_channel == f"{self.env_uuid}/{self.node_name}/observation":
                async with self.is_processing_observation_lock:
                    if self.is_processing_observation:
                        continue
                    self.is_processing_observation = True
                # Run the event handler in a separate task
                task = asyncio.create_task(
                    self.handle_event(input_channel, input_message)
                )
                self.tasks.append(task)
            else:
                await self.handle_event(input_channel, input_message)

        await asyncio.gather(*self.tasks)

    async def handle_event(self, input_channel, input_message):
        async for output_channel, output_message in self.event_handler(
            input_channel, input_message
        ):
            await self.r.publish(output_channel, output_message.model_dump_json())

    async def event_handler(
        self, input_channel: str, input_message: Message[JsonObj]
    ) -> AsyncIterator[tuple[str, Message[JsonObj]]]:
        """
        Process incoming messages and generate appropriate responses.

        Handles three types of messages:
        1. Start: Initialize the simulated user with task parameters
        2. End: Clean up resources and log results
        3. Observation: Generate and send appropriate user actions

        Args:
            input_channel: The Redis channel receiving the message
            input_message: The received message containing task data

        Returns:
            AsyncIterator yielding (channel, message) pairs for responses

        Raises:
            asyncio.CancelledError: When task completes or cleanup is needed
        """
        if input_channel == f"{self.env_uuid}/start":
            self.simulated_user.start(
                name=self.node_name,
                team_members=input_message.data.object["team_members"],
                task_description=input_message.data.object["task_description"],
                action_space=input_message.data.object["action_space"],
                private_information=input_message.data.object["additional_task_info"],
            )
        elif input_channel == f"{self.env_uuid}/end":
            self.simulated_user.end(result_dir=input_message.data.object["result_dir"])
            for task in self.tasks:
                task.cancel()
            await self.delete_process_record()
            raise asyncio.CancelledError
        elif input_channel == f"{self.env_uuid}/{self.node_name}/observation":
            observation = input_message.data.object["observation"]
            chat_history = input_message.data.object["chat_history"]
            action = self.simulated_user.get_action(
                observation=observation, chat_history=chat_history
            )
            payload = {"action": action, "role": self.node_name}
            await self.update_last_active_time()
            yield f"{self.env_uuid}/step", Message[JsonObj](
                data=JsonObj(object=payload)
            )
            async with self.is_processing_observation_lock:
                self.is_processing_observation = False
