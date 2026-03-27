

# 每一次使用前运行这两个命令
conda activate isaaclab2.3.2

source /home/ubuntu/workspace/isaac/isaac-sim/setup_conda_env.sh

# 训练与测试
python3 scripts/rsl_rl/train.py --task=Dirct-Pm01-v0 --num_envs=4096 --headless --video --video_length 200 --video_interval 10000 --max_iterations 6000

python3 scripts/rsl_rl/play.py --task=Dirct-Pm01-v0 --num_envs=8 --checkpoint 

python3 -m tensorboard.main --logdir 

./isaaclab.sh -p scripts/environments/export_IODescriptors.py --task=Isaac-Velocity-Flat-PM01-v0 --output_dir ./io_descriptors
