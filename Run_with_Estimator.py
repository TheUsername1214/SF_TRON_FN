import sys
import os
# 自动获取当前文件的目录，然后找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 假设脚本在项目子目录中
sys.path.insert(0, project_root)
from SF_TRON_FN.SRC.Env.TronEnv import TronEnv
from SF_TRON_FN.SRC.PPO.Actor_Critic import Actor_Critic
from SF_TRON_FN.SRC.Config.TS_Config import *
from SF_TRON_FN.SRC.Estimator.Estimator import *
from SF_TRON_FN.SRC.Plotter.ImagePlotter import *

Img = ImagePlotter(image_number=2)
PPO_1 = Actor_Critic(PPOCfg, EnvCfg, index=1)
Estimator_1 = Estimator(PPOCfg, EnvCfg, index=1)
Env = TronEnv(EnvCfg, RobotCfg, PPOCfg)
Env.prim_initialization(reset_all=True)

if not EnvCfg.EnvParam.train:
    PPO_1.load_best_model()
    Estimator_1.load_each_epi_model()


for epi in range(PPOCfg.PPOParam.episode):
    print(f"===================episode: {epi}===================")
    """每一段时间改个命令和加干扰"""
    if epi % int(5 / (EnvCfg.EnvParam.dt * PPOCfg.PPOParam.maximum_step) + 1) == 0:
        Env.resample_command()
        Env.apply_disturbance()
    state = Env.get_current_observations()
    Estimator_1.store_forward_state(state)
    for step in range(PPOCfg.PPOParam.maximum_step):
        """获取当前状态"""
        state = Env.get_current_observations()
        privilege_state = Env.get_privilege()
        estimated_privilege_state = Estimator_1.get_estimate_output()
        full_state = torch.concatenate((state, estimated_privilege_state), dim=-1)
        """不训练就画图"""
        if not EnvCfg.EnvParam.train:
            privilege_state = Env.get_privilege()
            est = Estimator_1.get_estimate_output()

            Img.append(epi * PPOCfg.PPOParam.maximum_step + step, 100 * est[:, 7:8][0, 0].item(), 0)
            Img.append(epi * PPOCfg.PPOParam.maximum_step + step, 100 * privilege_state[:, 7:8][0, 0].item(), 1)
            # Img.animation_plot()

        """做动作"""
        action, scaled_action = PPO_1.sample_action(full_state, deterministic=not EnvCfg.EnvParam.train)

        """更新环境"""
        Env.update_world(scaled_action=scaled_action)

        """获取下一个状态"""

        next_state = Env.get_next_observations()
        Estimator_1.store_forward_state(next_state)
        next_estimated_privilege_state = Estimator_1.get_estimate_output()
        next_full_state = torch.concatenate((next_state, next_estimated_privilege_state), dim=-1)

        """计算奖励 判断是否结束"""

        reward, over, truncated = Env.compute_reward()

        """存储经验"""
        if EnvCfg.EnvParam.train:
            PPO_1.store_experience(full_state,
                                   action,
                                   next_full_state,
                                   reward,
                                   over,
                                   step)

            Estimator_1.store_new_state_and_output(state,
                                                   privilege_state,
                                                   step,
                                                   over)

        """重置挂掉的机器人"""
        over += truncated
        Env.prim_initialization(torch.nonzero(over.flatten()).flatten())

    """每个回合结束后训练一次"""
    if EnvCfg.EnvParam.train:
        PPO_1.update()
        Estimator_1.update()
        Env.print_reward_sum()
