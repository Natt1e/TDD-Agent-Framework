# run reflect
python run_repoeval_tdd.py \
--traj-output your_traj/ \
--config config/ablation/tdd_agent_reflect.yaml \
--predict-output your_predict.jsonl \
--docker-cpus 2.0 \
--docker-memory 4g 


# run single track
python run_repoeval_tdd.py \
--traj-output your_traj/ \
--config config/ablation/tdd_agent_single_track.yaml \
--predict-output your_predict.jsonl \
--docker-cpus 2.0 \
--docker-memory 4g 


# run vanilla
python run_repoeval_tdd.py \
--traj-output your_traj/ \
--config config/ablation/tdd_agent_vanilla.yaml \
--predict-output your_predict.jsonl \
--docker-cpus 2.0 \
--docker-memory 4g


