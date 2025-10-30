MODEL_NAME="claude40"

START_IDX=0
END_IDX=102

python -m scripts.fully_auto_agent_exp \
    --task "travel_planning" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --team-member-config-path configs/teams/auto_agent_team_config_${MODEL_NAME}.toml \
    --result-dir-tag travel_planning_auto_agent_${MODEL_NAME}

python -m scripts.collaborative_agent_exp \
    --task "travel_planning" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --team-member-config-path configs/teams/basic_coagent_simulated_user_team_config_${MODEL_NAME}.toml \
    --result-dir-tag travel_planning_basic_coagent_${MODEL_NAME}

python -m scripts.collaborative_agent_exp \
    --task "travel_planning" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --team-member-config-path configs/teams/coagent_with_situational_planning_simulated_user_team_config_${MODEL_NAME}.toml \
    --result-dir-tag travel_planning_coagent_with_situational_planning_${MODEL_NAME}