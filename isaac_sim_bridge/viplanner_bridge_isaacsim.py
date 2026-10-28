"""Isaac Sim port of legged_gym/envs/go2_stairs/viplanner_bridge.py.

Identical to the original except the single dependency on isaacgym.torch_utils
(quat_apply / quat_conjugate, both xyzw-convention) is replaced with an inline,
byte-for-byte reimplementation of the same two functions (this is the standard
Isaac Gym torch_utils formula, xyzw quaternions) so this file has zero
Isaac-Gym dependency. Everything else -- load_model, goal_transformer,
path_transformer, input_transformer, plan/plan_dual -- is unchanged.
"""
import os

import torch
import torch.nn.functional as F

from viplanner.config import TrainCfg
from viplanner.plannernet import AutoEncoder, DualAutoEncoder
from viplanner.traj_cost_opt.traj_opt import TrajOpt


def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, 3:]), dim=-1).view(shape)


def quat_apply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


def _resize(img: torch.Tensor, size) -> torch.Tensor:
    if tuple(img.shape[-2:]) == tuple(size):
        return img
    return F.interpolate(img, size=tuple(size), mode="bilinear", align_corners=False)


class VIPlannerAlgo:
    def __init__(self, model_dir: str, fear_threshold: float = 0.5, device: str = "cuda"):
        assert os.path.exists(model_dir), "Model directory does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.pt")), "Model file does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.yaml")), "Model config file does not exist"

        self.fear_threshold = fear_threshold
        self.device = device

        self.train_config: TrainCfg = None
        self.load_model(model_dir)

        self.transform = lambda img: _resize(img, self.train_config.img_input_size)
        self.traj_generate = TrajOpt()

    def load_model(self, model_dir: str):
        self.train_config = TrainCfg.from_yaml(os.path.join(model_dir, "model.yaml"))
        print(
            f"[VIPlannerAlgo] loaded model using sem={self.train_config.sem}, rgb={self.train_config.rgb}, "
            f"knodes={self.train_config.knodes}, in_channel={self.train_config.in_channel}"
        )

        if isinstance(self.train_config.data_cfg, list):
            self.max_goal_distance = self.train_config.data_cfg[0].max_goal_distance
            self.max_depth = self.train_config.data_cfg[0].max_depth
        else:
            self.max_goal_distance = self.train_config.data_cfg.max_goal_distance
            self.max_depth = self.train_config.data_cfg.max_depth

        if self.train_config.sem:
            self.net = DualAutoEncoder(self.train_config)
        else:
            self.net = AutoEncoder(self.train_config.in_channel, self.train_config.knodes)

        try:
            model_state_dict, _ = torch.load(os.path.join(model_dir, "model.pt"), weights_only=True)
        except (ValueError, RuntimeError):
            model_state_dict, _ = torch.load(os.path.join(model_dir, "model.pt"), weights_only=False)
        self.net.load_state_dict(model_state_dict)
        self.net.eval()

        if self.device.lower() == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
            self.cuda_avail = False
        else:
            self.net = self.net.to(self.device)
            self.cuda_avail = True

    ###
    # Transformations
    ###

    def goal_transformer(self, goal: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> torch.Tensor:
        """transform goal into camera frame"""
        goal_cam_frame = goal - cam_pos
        goal_cam_frame = goal_cam_frame.clone()
        goal_cam_frame[:, 2] = 0  # trained with z difference of 0
        goal_cam_frame = quat_apply(quat_conjugate(cam_quat), goal_cam_frame)
        return goal_cam_frame

    def path_transformer(self, path_cam_frame: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> torch.Tensor:
        """transform path from camera frame to world frame"""
        return quat_apply(cam_quat.unsqueeze(1).repeat(1, path_cam_frame.shape[1], 1), path_cam_frame) + cam_pos.unsqueeze(1)

    def input_transformer(self, image: torch.Tensor) -> torch.Tensor:
        image = self.transform(image)
        image = image.clone()
        image[image > self.max_depth] = 0.0
        image[~torch.isfinite(image)] = 0
        return image

    ###
    # Planning
    ###

    def plan(self, image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        with torch.no_grad():
            keypoints, fear = self.net(self.input_transformer(image), goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        return keypoints, traj, fear

    def plan_dual(self, dep_image: torch.Tensor, sem_image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        sem_image = self.transform(sem_image) / 255
        with torch.no_grad():
            keypoints, fear = self.net(self.input_transformer(dep_image), sem_image, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        return keypoints, traj, fear
