import json
import os
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from knowledge_storm import GoogleSearch, AzureOpenAIModel

from collaborative_gym.core import CoEnv, ObservationTypes, logger
from collaborative_gym.envs.registry import EnvFactory
from collaborative_gym.envs.travel_planner.evaluation import (
    HardConstraintEvaluator,
    CommonsenseConstraintEvaluator,
)
from collaborative_gym.envs.travel_planner.tools import (
    Flights,
    Accommodations,
    Restaurants,
    Attractions,
    GoogleDistanceMatrix,
)
from collaborative_gym.spaces import (
    MultiSpace,
    MAX_UNICODE_LENGTH,
    UnicodeWithRegexPattern,
)
from collaborative_gym.utils.business_search import GooglePlace
from collaborative_gym.utils.map import GoogleMap
from collaborative_gym.utils.string import post_process_parsed_function_arg
from collaborative_gym.utils.text_editor import TextEditor


class CoTravelPlanningActions(Enum):
    DISTANCE_MATRIX = "DISTANCE_MATRIX"
    EDITOR_UPDATE = "EDITOR_UPDATE"
    FINISH = "FINISH"
    # For Co-Gym (real)
    BUSINESS_SEARCH = "BUSINESS_SEARCH"
    INTERNET_SEARCH = "INTERNET_SEARCH"
    # For Co-Gym (simulated) using TravelPlanner dataset for this task
    FLIGHT_SEARCH = "FLIGHT_SEARCH"
    ACCOMMODATION_SEARCH = "ACCOMMODATION_SEARCH"
    RESTAURANT_SEARCH = "RESTAURANT_SEARCH"
    ATTRACTION_SEARCH = "ATTRACTION_SEARCH"

    def __str__(self):
        return self.value


@EnvFactory.register("travel_planning")
class CoTravelPlanningEnv(CoEnv):
    """
    ## Description
    CoTravelPlanningEnv is a collaborative environment for planning travel itineraries.
    The environment supports several search functionalities, computing distances between destinations, and editing travel plans.

    ## Action Space
    Actions are strings that must match one of the following patterns:
    - Search functions in Co-Gym (real):
        - BUSINESS_SEARCH(query: str, location: str, limit: int): Search for businesses
        - INTERNET_SEARCH(query: str): Perform web searches
    - Search functions in Co-Gym (simulated):
        - FLIGHT_SEARCH(origin: str, destination: str, date: str): Find flights
        - ACCOMMODATION_SEARCH(city: str): Find lodging
        - RESTAURANT_SEARCH(city: str): Find dining
        - ATTRACTION_SEARCH(city: str): Find attractions
    - DISTANCE_MATRIX(origins: str, destinations: str): Calculate travel times and distances
    - EDITOR_UPDATE(text: str): Update editor content
    - FINISH(): Complete task

    ## Observation Space
    The observation is a dictionary containing:
    - travel_plan_editor (non-private): Current state of the travel plan
    - search_output (non-private): Results from recent searches
    - distance_matrix_output (non-private): Latest travel time calculations
    """

    def __init__(
        self,
        team_members: List[str],
        env_id: str,
        use_simulated_dataset: bool = False,
        query: Optional[str] = None,
        travel_planner_data_point_idx: Optional[int] = None,  # [0, 1, ..., 101]
        travel_planner_data_path: str = "datasets/TravelPlanner/validation_with_hidden_profile.csv",
        travel_planner_database_dir: str = "datasets/TravelPlanner/database",
    ):
        super().__init__(team_members=team_members, env_id=env_id)

        self.use_simulated_dataset = use_simulated_dataset
        self.travel_planner_database_dir = travel_planner_database_dir

        if self.use_simulated_dataset:
            self.travel_planner_data_point_idx = travel_planner_data_point_idx
            travel_planner_validation_set = pd.read_csv(travel_planner_data_path)
            query_template = "Help me plan a {days}-day trip from {org} to {dest} starting on {start_date}."
            self.query = query_template.format(
                days=travel_planner_validation_set.iloc[travel_planner_data_point_idx][
                    "days"
                ],
                org=travel_planner_validation_set.iloc[travel_planner_data_point_idx][
                    "org"
                ],
                dest=travel_planner_validation_set.iloc[travel_planner_data_point_idx][
                    "dest"
                ],
                start_date=eval(
                    travel_planner_validation_set.iloc[travel_planner_data_point_idx][
                        "date"
                    ]
                )[0],
            )
            self.additional_task_info = {
                "Preferences and constraints to stick to": eval(
                    travel_planner_validation_set.iloc[travel_planner_data_point_idx][
                        "preferences"
                    ]
                ),
            }
            try:
                self.parsing_lm = AzureOpenAIModel(
                    model="gpt-4o-2024-08-06",
                    api_key=os.environ["AZURE_API_KEY"],
                    azure_endpoint=os.environ["AZURE_ENDPOINT"],
                    api_version=os.environ["AZURE_API_VERSION"],
                )
            except KeyError:
                self.parsing_lm = None
                logger.error(
                    "Please provide your OpenAI API key in the environment variable OPENAI_API_KEY to enable the evaluator."
                )
            self.travel_planner_commonsense_evaluator = CommonsenseConstraintEvaluator(
                database_dir=self.travel_planner_database_dir
            )
            self.travel_planner_preference_evaluator = HardConstraintEvaluator(
                database_dir=self.travel_planner_database_dir
            )
            self.travel_planner_ground_truth = travel_planner_validation_set.iloc[
                travel_planner_data_point_idx
            ].to_dict()
            self.travel_planner_ground_truth["local_constraint"] = eval(
                self.travel_planner_ground_truth["local_constraint"]
            )
        else:
            self.query = query

        # Task information
        self.task_description = (
            "You are a proficient travel planner. Your task is to plan a detailed travel "
            "itinerary for a user or a group of users. Note that all the information in your "
            "plan should be grounded in searched information. The plan should also align "
            "with commonsense.\n"
            f"Here is the initial query: {self.query}\n"
            "Edit the travel plan for the query in the editor. Note that the editor is in a "
            "shared workbench which means it can be viewed and edited by every team member. Your "
            "change to the editor will also be shown to the team member in real time so you "
            "don't need to send the content to them. Once finished, you performance will be "
            "evaluated based on the feasibility and user satisfaction of the plan in the "
            "editor.\n"
        )
        if self.use_simulated_dataset:
            # Additional task description to specify the plan format that can be parsed by TravelPlanner evaluation script
            self.task_description += (
                "Specifically, the detailed plan should include specific information such as "
                "flight numbers (e.g., F0123456), restaurant names, and accommodation names.\n"
                "Here is an example for what the final plan shall look like (you need to "
                "strictly follow the format when writing the travel plan):\n"
                "Query: Could you create a travel plan for 7 people from Ithaca to Charlotte "
                "spanning 3 days, from March 8th to March 10th, 2022, with a budget of $30,200?\n"
                "Travel Plan:\n"
                "Day 1:\nCurrent City: from Ithaca to Charlotte\n"
                "Transportation: Flight Number: F3633413, from Ithaca to Charlotte, "
                "Departure Time: 05:38, Arrival Time: 07:46\nBreakfast: Nagaland's Kitchen, "
                "Charlotte\nAttraction: The Charlotte Museum of History, Charlotte\n"
                "Lunch: Cafe Maple Street, Charlotte\nDinner: Bombay Vada Pav, Charlotte\n"
                "Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte\n\n"
                "Day 2:\nCurrent City: Charlotte\nTransportation: -\n"
                "Breakfast: Olive Tree Cafe, Charlotte\n"
                "Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.\n"
                "Lunch: Birbal Ji Dhaba, Charlotte\nDinner: Pind Balluchi, Charlotte\n"
                "Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte\n\n"
                "Day 3:\nCurrent City: from Charlotte to Ithaca\n"
                "Transportation: Flight Number: F3786167, from Charlotte to Ithaca, "
                "Departure Time: 21:42, Arrival Time: 23:26\n"
                "Breakfast: Subway, Charlotte\nAttraction: Books Monument, Charlotte.\n"
                "Lunch: Olive Tree Cafe, Charlotte\nDinner: Kylin Skybar, Charlotte\n"
                "Accommodation: -"
            )
            # An example question and trajectory for team members to understand the task
            self.example_question = (
                "Could you create a travel plan for 7 people from Ithaca to Charlotte panning 3 "
                "days, from March 8th to March 10th, 2022, with a budget of $30,200?"
            )
            self.example_trajectory = [
                (
                    "First, search for flights from Ithaca to Charlotte on March 8th, 2022.",
                    "FLIGHT_SEARCH(origin='Ithaca', destination='Charlotte', date='2022-03-08')",
                    {
                        "search_output": {
                            "query": "Flight Search (origin: Ithaca, destination: Charlotte, date: 2022-03-08)",
                            "output": """        Flight Number  Price DepTime ArrTime   ActualElapsedTime  FlightDate OriginCityName DestCityName  Distance
3403495      F3633405    228   05:38   08:10  2 hours 32 minutes  2022-03-08         Ithaca    Charlotte     557.0""",
                        }
                    },
                ),
                (
                    "Next, search for accommodations in Charlotte.",
                    "ACCOMMODATION_SEARCH(city='Charlotte')",
                    {
                        "search_output": {
                            "query": "Accommodation Search (city: Charlotte)",
                            "output": """                                                   NAME   price        room type  ... maximum occupancy  review rate number       city
56                                 Cute Greenpoint room   335.0     Private room  ...                 1                 2.0  Charlotte
568              Elegant Studio Apt in Prospect Heights   640.0  Entire home/apt  ...                 5                 5.0  Charlotte
613                    Two Bedrooms with Dedicated Bath   615.0     Private room  ...                 2                 4.0  Charlotte
776                 Lower East Side 1bedroom apt in NYC   861.0  Entire home/apt  ...                 6                 3.0  Charlotte""",
                        }
                    },
                ),
                (
                    "...(omitted several steps)... Now, I have gathered several information. I can update the editor first",
                    "EDITOR_UPDATE(text='Day 1:\\nCurrent City: from Ithaca to Charlotte\\n"
                    "Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, "
                    "Arrival Time: 07:46\\nBreakfast: Nagaland's Kitchen, Charlotte\\nAttraction: The Charlotte "
                    "Museum of History, Charlotte\\nLunch: Cafe Maple Street, Charlotte\\nDinner: Bombay Vada Pav, "
                    "Charlotte\\nAccommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte')",
                    {
                        "travel_plan_editor": "Day 1:\nCurrent City: from Ithaca to Charlotte\nTransportation: "
                        "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: "
                        "05:38, Arrival Time: 07:46\nBreakfast: Nagaland's Kitchen, Charlotte\n"
                        "Attraction: The Charlotte Museum of History, Charlotte\nLunch: Cafe "
                        "Maple Street, Charlotte\nDinner: Bombay Vada Pav, Charlotte\n"
                        "Accommodation: Affordable Spacious Refurbished Room in Bushwick!, "
                        "Charlotte"
                    },
                ),
            ]

        else:
            # An example question and trajectory for team members to understand the task
            self.example_question = (
                "Could you create a travel plan for 7 people from Ithaca to Charlotte panning 3 "
                "days, from March 8th to March 10th, 2022?"
            )
            self.example_trajectory = [
                (
                    "First, search for accommodations in Charlotte.",
                    "BUSINESS_SEARCH(term='hotels', location='Charlotte', limit=5)",
                    {
                        "search_output": {
                            "query": "Business Search (term: hotels, location: Charlotte, limit: 5)",
                            "output": """                                                   NAME   price        room type  ... maximum occupancy  review rate number       city
            56                                 Cute Greenpoint room   335.0     Private room  ...                 1                 2.0  Charlotte
            568              Elegant Studio Apt in Prospect Heights   640.0  Entire home/apt  ...                 5                 5.0  Charlotte
            613                    Two Bedrooms with Dedicated Bath   615.0     Private room  ...                 2                 4.0  Charlotte
            776                 Lower East Side 1bedroom apt in NYC   861.0  Entire home/apt  ...                 6                 3.0  Charlotte""",
                        }
                    },
                ),
                (
                    "...(omitted several steps)... Now, I have gathered several information. I can update the editor first",
                    "EDITOR_UPDATE(text='## Day 1:\\nCurrent City: from Ithaca to Charlotte\\n"
                    "**Attraction:** The Charlotte Museum of History, Charlotte\\n**Lunch:** Cafe Maple Street, "
                    "Charlotte\\n**Dinner:** Bombay Vada Pav, "
                    "Charlotte\\n**Accommodation:** Affordable Spacious Refurbished Room in Bushwick!, Charlotte')",
                    {
                        "travel_plan_editor": "## Day 1:\\nCurrent City: from Ithaca to Charlotte\\n**Attraction:** "
                        "The Charlotte Museum of History, Charlotte\\n**Lunch:** Cafe Maple "
                        "Street, Charlotte\\n**Dinner:** Bombay Vada Pav, Charlotte\\n"
                        "**Accommodation:** Affordable Spacious Refurbished Room in Bushwick!, "
                        "Charlotte"
                    },
                ),
            ]

        # Action Space
        self.action_space = MultiSpace(
            (
                UnicodeWithRegexPattern(
                    min_length=0,
                    max_length=MAX_UNICODE_LENGTH,
                    regex_pattern=re.compile(
                        r"^EDITOR_UPDATE\(text=(.*)\)$", re.DOTALL
                    ),
                    params=["text"],
                    machine_readable_identifier=CoTravelPlanningActions.EDITOR_UPDATE,
                    human_readable_name="Update the travel plan editor",
                    human_readable_description="Update the travel plan editor with the provided text. The full original"
                    " text will be replaced.",
                ),
                UnicodeWithRegexPattern(
                    min_length=0,
                    max_length=MAX_UNICODE_LENGTH,
                    regex_pattern=re.compile(r"^FINISH\(\)$", re.DOTALL),
                    params=[],
                    machine_readable_identifier=CoTravelPlanningActions.FINISH,
                    human_readable_name="Finish the travel planning task",
                    human_readable_description="Finish the travel planning task.",
                ),
            )
        )

        if self.use_simulated_dataset:
            # Load simulated search tools
            self.flight_search = Flights(
                path=os.path.join(
                    self.travel_planner_database_dir, "flights/clean_Flights_2022.csv"
                )
            )
            self.accommodation_search = Accommodations(
                path=os.path.join(
                    self.travel_planner_database_dir,
                    "accommodations/clean_accommodations_2022.csv",
                )
            )
            self.restaurant_search = Restaurants(
                path=os.path.join(
                    self.travel_planner_database_dir,
                    "restaurants/clean_restaurant_2022.csv",
                )
            )
            self.google_distance_matrix = GoogleDistanceMatrix(
                path=os.path.join(
                    self.travel_planner_database_dir,
                    "googleDistanceMatrix/distance.csv",
                )
            )
            self.attraction_search = Attractions(
                path=os.path.join(
                    self.travel_planner_database_dir, "attractions/attractions.csv"
                )
            )
            self.private_action_space = MultiSpace(
                (
                    UnicodeWithRegexPattern(
                        min_length=0,
                        max_length=MAX_UNICODE_LENGTH,
                        regex_pattern=re.compile(
                            r"^FLIGHT_SEARCH\(origin=(.*), destination=(.*), date=(.*)\)$",
                            re.DOTALL,
                        ),
                        params=["origin", "destination", "date"],
                        machine_readable_identifier=CoTravelPlanningActions.FLIGHT_SEARCH,
                        human_readable_name="Search for flights",
                        human_readable_description="Search for flights based on the provided origin, destination, and "
                        "date.",
                    ),
                    UnicodeWithRegexPattern(
                        min_length=0,
                        max_length=MAX_UNICODE_LENGTH,
                        regex_pattern=re.compile(
                            r"^ACCOMMODATION_SEARCH\(city=(.*)\)$", re.DOTALL
                        ),
                        params=["city"],
                        machine_readable_identifier=CoTravelPlanningActions.ACCOMMODATION_SEARCH,
                        human_readable_name="Search for accommodations",
                        human_readable_description="Search for accommodations within a certain city.",
                    ),
                    UnicodeWithRegexPattern(
                        min_length=0,
                        max_length=MAX_UNICODE_LENGTH,
                        regex_pattern=re.compile(
                            r"^RESTAURANT_SEARCH\(city=(.*)\)$", re.DOTALL
                        ),
                        params=["city"],
                        machine_readable_identifier=CoTravelPlanningActions.RESTAURANT_SEARCH,
                        human_readable_name="Search for restaurants",
                        human_readable_description="Search for restaurants within a certain city.",
                    ),
                    UnicodeWithRegexPattern(
                        min_length=0,
                        max_length=MAX_UNICODE_LENGTH,
                        regex_pattern=re.compile(
                            r"^ATTRACTION_SEARCH\(city=(.*)\)$", re.DOTALL
                        ),
                        params=["city"],
                        machine_readable_identifier=CoTravelPlanningActions.ATTRACTION_SEARCH,
                        human_readable_name="Search for attractions",
                        human_readable_description="Search for attractions within a certain city.",
                    ),
                    UnicodeWithRegexPattern(
                        min_length=0,
                        max_length=MAX_UNICODE_LENGTH,
                        regex_pattern=re.compile(
                            r"^DISTANCE_MATRIX\(origins=(.*), destinations=(.*)\)$",
                            re.DOTALL,
                        ),
                        params=["origin", "destination"],
                        machine_readable_identifier=CoTravelPlanningActions.DISTANCE_MATRIX,
                        human_readable_name="Get distance matrix from Google Maps",
                        human_readable_description="Get distance matrix (based on the driving time) from Google Maps "
                        "based on the provided origins and destinations.",
                    ),
                )
            )
        else:
            # Use real search tools
            try:
                self.business_search = GooglePlace(
                    google_map_api_key=os.environ["GOOGLE_MAP_API_KEY"]
                )
                self.google_map = GoogleMap(
                    google_map_api_key=os.environ["GOOGLE_MAP_API_KEY"]
                )
                self.search_engine = GoogleSearch(
                    google_search_api_key=os.environ["GOOGLE_SEARCH_API_KEY"],
                    google_cse_id=os.environ["GOOGLE_CSE_ID"],
                    k=10,
                )
            except KeyError:
                self.business_search = None
                self.google_map = None
                self.search_engine = None
                logger.error(
                    "Please provide your Google API key in the environment variable GOOGLE_MAP_API_KEY and "
                    "GOOGLE_SEARCH_API_KEY, GOOGLE_CSE_ID to enable the search functionalities."
                )

            self.private_action_space = MultiSpace(
                (
                    UnicodeWithRegexPattern(
                        min_length=0,
                        max_length=MAX_UNICODE_LENGTH,
                        regex_pattern=re.compile(
                            r"^BUSINESS_SEARCH\(term=(.*), location=(.*), limit=(.*)\)$",
                            re.DOTALL,
                        ),
                        params=["term", "location", "limit"],
                        machine_readable_identifier=CoTravelPlanningActions.BUSINESS_SEARCH,
                        human_readable_name="Find businesses (restaurants, hotels, etc.) on Google Places",
                        human_readable_description="Find businesses (restaurants, hotels, etc.) on Google Places based on "
                        "the provided term, location, and limit. term and location are strings, "
                        "and limit is an integer.",
                    ),
                    UnicodeWithRegexPattern(
                        min_length=0,
                        max_length=MAX_UNICODE_LENGTH,
                        regex_pattern=re.compile(
                            r"^DISTANCE_MATRIX\(origins=(.*), destinations=(.*), mode=(.*)\)$",
                            re.DOTALL,
                        ),
                        params=["origins", "destinations", "mode"],
                        machine_readable_identifier=CoTravelPlanningActions.DISTANCE_MATRIX,
                        human_readable_name="Get distance matrix from Google Maps",
                        human_readable_description="Get distance matrix from Google Maps based on the provided origins, "
                        "destinations, and mode. origins and destinations are lists of strings, "
                        'and mode can be "driving", "walking", "bicycling", or "transit".',
                    ),
                    UnicodeWithRegexPattern(
                        min_length=0,
                        max_length=MAX_UNICODE_LENGTH,
                        regex_pattern=re.compile(
                            r"^INTERNET_SEARCH\(query=(.*)\)$", re.DOTALL
                        ),
                        params=["query"],
                        machine_readable_identifier=CoTravelPlanningActions.INTERNET_SEARCH,
                        human_readable_name="Search the Internet",
                        human_readable_description="Search the Internet based on the provided query.",
                    ),
                )
            )

        # Private observation
        self.search_output = {team_member: {} for team_member in team_members}

        self.distance_matrix_output = {team_member: {} for team_member in team_members}

        # Shared observation
        self.travel_plan_editor = TextEditor()

    # Private actions (Co-Gym real)
    def _business_search(self, role: str, term: str, location: str, limit: int):
        output = self.business_search.search(term=term, location=location, limit=limit)
        self.search_output[role] = {
            "query": f"Business Search (term: {term}, location: {location}, limit: {limit})",
            "output": output,
        }

    def _distance_matrix(self, role: str, origins: str, destinations: str, mode: str):
        origins_list = json.loads(origins)
        destinations_list = json.loads(destinations)
        output = self.google_map.distance_matrix(
            origins=origins_list, destinations=destinations_list, mode=mode
        )
        self.distance_matrix_output[role] = {
            "query": f"Distance Matrix (origins: {origins}, destinations: {destinations}, mode: {mode})",
            "output": output,
        }

    def _internet_search(self, role: str, query: str):
        output = self.search_engine(query)
        self.search_output[role] = {
            "query": f"Internet Search (query: {query})",
            "output": output,
        }

    # Private actions (Co-Gym simulated)
    def _flight_search(self, role: str, origin: str, destination: str, date: str):
        output = self.flight_search.run(
            origin=origin, destination=destination, departure_date=date
        )
        if type(output) == pd.DataFrame:
            output = output.to_string()
        self.search_output[role] = {
            "query": f"Flight Search (origin: {origin}, destination: {destination}, date: {date})",
            "output": output,
        }

    def _accommodation_search(self, role: str, city: str):
        output = self.accommodation_search.run(city=city)
        if type(output) == pd.DataFrame:
            output = output.to_string()
        self.search_output[role] = {
            "query": f"Accommodation Search (city: {city})",
            "output": output,
        }

    def _restaurant_search(self, role: str, city: str):
        output = self.restaurant_search.run(city=city)
        if type(output) == pd.DataFrame:
            output = output.to_string()
        self.search_output[role] = {
            "query": f"Restaurant Search (city: {city})",
            "output": output,
        }

    def _attraction_search(self, role: str, city: str):
        output = self.attraction_search.run(city=city)
        if type(output) == pd.DataFrame:
            output = output.to_string()
        self.search_output[role] = {
            "query": f"Attraction Search (city: {city})",
            "output": output,
        }

    def _distance_matrix_simulated(self, role: str, origin: str, destination: str):
        output = self.google_distance_matrix.run(origin=origin, destination=destination)
        self.distance_matrix_output[role] = {
            "query": f"Distance Matrix (origins: {origin}, destinations: {destination})",
            "output": output,
        }

    # Shared actions
    def _editor_update(self, text: str):
        self.travel_plan_editor.update_text(text)

    def close(self):
        pass

    def get_obs(self):
        return {
            "public": {
                "travel_plan_editor": self.travel_plan_editor.get_text(),
            },
            "private": {
                team_member: {
                    "search_output": self.search_output[team_member],
                    "distance_matrix_output": self.distance_matrix_output[team_member],
                }
                for team_member in self.team_members
            },
        }

    def obs_type(self) -> dict[str, ObservationTypes]:
        return {
            "travel_plan_editor": ObservationTypes.TEXT_EDITOR,
            "search_output": ObservationTypes.TRAVEL_SEARCH,
            "distance_matrix_output": ObservationTypes.DISTANCE_MATRIX,
        }

    def reset(
        self,
        options: dict[str, Any] | None = None,
    ):
        self.travel_plan_editor.update_text("")
        self.search_output = {team_member: {} for team_member in self.team_members}
        self.distance_matrix_output = {
            team_member: {} for team_member in self.team_members
        }

        return self.get_obs(), {}

    def step(self, role: str, action: str):
        """Execute one timestep within the environment.

        Args:
            role (str): The team member executing the action
            action (str): The action to take, formatted as a string matching one of the action space patterns

        Returns:
            observation (dict): Environment's current state
            reward (float): Amount of reward returned (-1 for errors, 0 otherwise)
            terminated (bool): Whether the episode has ended
            private (bool): Whether the action results are private to the acting agent
            info (dict): Contains auxiliary diagnostic information
        """
        info = {}
        info["action_start_time"] = time.time()

        # Parse and validate action using parent class helper
        parsed_action, private, action_id, err_msg = self.parse_and_validate_action(
            role, action
        )
        if err_msg:
            return self.handle_action_error(err_msg, private)

        # Post-process parsed action parameters
        for k in parsed_action:
            parsed_action[k] = post_process_parsed_function_arg(parsed_action[k])

        info["action"] = action_id

        # Execute the action
        terminated = False
        reward = 0  # Set intermediate reward to 0 if the action is successful; otherwise, -1.
        info["action_error"] = None
        try:
            if info["action"] == CoTravelPlanningActions.BUSINESS_SEARCH:
                self._business_search(
                    role=role,
                    term=parsed_action["term"],
                    location=parsed_action["location"],
                    limit=int(parsed_action["limit"]),
                )
            elif info["action"] == CoTravelPlanningActions.DISTANCE_MATRIX:
                if self.use_simulated_dataset:
                    self._distance_matrix_simulated(
                        role=role,
                        origin=parsed_action["origin"],
                        destination=parsed_action["destination"],
                    )
                else:
                    self._distance_matrix(
                        role=role,
                        origins=parsed_action["origins"],
                        destinations=parsed_action["destinations"],
                        mode=parsed_action["mode"],
                    )
            elif info["action"] == CoTravelPlanningActions.INTERNET_SEARCH:
                self._internet_search(role=role, query=parsed_action["query"])
            elif info["action"] == CoTravelPlanningActions.FLIGHT_SEARCH:
                self._flight_search(
                    role=role,
                    origin=parsed_action["origin"],
                    destination=parsed_action["destination"],
                    date=parsed_action["date"],
                )
            elif info["action"] == CoTravelPlanningActions.ACCOMMODATION_SEARCH:
                self._accommodation_search(role=role, city=parsed_action["city"])
            elif info["action"] == CoTravelPlanningActions.RESTAURANT_SEARCH:
                self._restaurant_search(role=role, city=parsed_action["city"])
            elif info["action"] == CoTravelPlanningActions.ATTRACTION_SEARCH:
                self._attraction_search(role=role, city=parsed_action["city"])
            elif info["action"] == CoTravelPlanningActions.EDITOR_UPDATE:
                self._editor_update(text=parsed_action["text"])
            elif info["action"] == CoTravelPlanningActions.FINISH:
                terminated = True
        except Exception as e:
            err_msg = f"Error in executing the action: {action}. Error: {e}"
            return self.handle_action_error(err_msg, private)
        finally:
            info["action_end_time"] = time.time()

        # Get the observation
        obs = self.get_obs()

        return obs, reward, terminated, private, info

    def eval_helper_parse_travel_plan(self, travel_plan: str) -> Dict:
        """Adapted from https://github.com/OSU-NLP-Group/TravelPlanner/blob/main/postprocess/openai_request.py
        `build_plan_format_conversion_prompt`"""

        prompt = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B). Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'. If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols.
-----EXAMPLE-----
 [{{
        "days": 1,
        "current_city": "from Dallas to Peoria",
        "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
        "breakfast": "-",
        "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
        "lunch": "-",
        "dinner": "Tandoor Ka Zaika, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 2,
        "current_city": "Peoria",
        "transportation": "-",
        "breakfast": "Tandoor Ka Zaika, Peoria",
        "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
        "lunch": "Cafe Hashtag LoL, Peoria",
        "dinner": "The Curzon Room - Maidens Hotel, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 3,
        "current_city": "from Peoria to Dallas",
        "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-"
    }}]
-----EXAMPLE END-----"""
        prompt += f"\nText:\n{travel_plan}\nJSON:\n"
        output = self.parsing_lm(prompt, temperature=0, max_tokens=4000)[0]
        if "```json" in output:
            output = output.split("```json")[1].strip("`").strip()

        try:
            parsed_results = eval(output)
            return parsed_results
        except Exception as e:
            print(f"Error: {e}")
            return {"output": output}

    def evaluate_task_performance(self) -> Dict:
        return self._evaluate_task_performance(
            self.travel_plan_editor.get_text()
        )
    
    def _evaluate_task_performance(self, _text) -> Dict:
        performance = {"outcome": _text}
        if len(_text) == 0:
            performance["task_completion"] = 0
            performance["performance_rating"] = 0
            return performance
        else:
            performance["task_completion"] = 1
        if self.use_simulated_dataset:
            performance["idx"] = self.travel_planner_data_point_idx
            performance["parsed_travel_plan"] = self.eval_helper_parse_travel_plan(
                _text
            )
            try:
                performance["commonsense_evaluation"] = (
                    self.travel_planner_commonsense_evaluator.run(
                        self.travel_planner_ground_truth,
                        performance["parsed_travel_plan"],
                    )
                )
            except Exception as e:
                performance["commonsense_evaluation"] = {}
            try:
                performance["preference_evaluation"] = (
                    self.travel_planner_preference_evaluator.run(
                        self.travel_planner_ground_truth,
                        performance["parsed_travel_plan"],
                    )
                )
            except Exception as e:
                performance["preference_evaluation"] = {}
            commonsense_pass_cnt = 0
            commonsense_criteria_cnt = 0
            for k, v in performance["commonsense_evaluation"].items():
                if v[0] is None:
                    continue
                if v[0]:
                    commonsense_pass_cnt += 1
                commonsense_criteria_cnt += 1
            preference_pass_cnt = 0
            preference_criteria_cnt = 0
            for k, v in performance["preference_evaluation"].items():
                if v[0] is None:
                    continue
                if v[0]:
                    preference_pass_cnt += 1
                preference_criteria_cnt += 1
            performance["commonsense_pass_rate"] = (
                commonsense_pass_cnt / commonsense_criteria_cnt
                if commonsense_criteria_cnt > 0
                else 0
            )
            performance["preference_pass_rate"] = (
                preference_pass_cnt / preference_criteria_cnt
                if preference_criteria_cnt > 0
                else 0
            )
            performance["performance_rating"] = (
                performance["commonsense_pass_rate"]
                + performance["preference_pass_rate"]
            ) / 2
        else:
            performance["query"] = self.query

        return performance

    def __repr__(self):
        return f"CoTravelPlanningEnv(use_simulated_dataset={self.use_simulated_dataset}, query={self.query})"
