from pathlib import Path

import torch


class ManipulationCfg:
    class ActionCfg:
        num_actions: int = 6
        dof_clip_actions: float = 10.0
        action_scale: float = 0.25

    class ObsCfg:
        class ObsScaleCfg:
            joint_pos: float = 1.0
            joint_vel: float = 1.0
            last_action: float = 1.0

        obs_history_length: int = 30
        num_single_observations: int = 27
        num_observations: int = (
            num_single_observations * obs_history_length
        )
        obs_scale: ObsScaleCfg = ObsScaleCfg()

    class PolicyCfg:
        hidden_dims: tuple[int, int, int] = (512, 256, 128)
        activation: str = "elu"


class ManipulationPolicy(torch.nn.Module):
    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.cfg = ManipulationCfg()
        self.device = torch.device(device)

        observation_dim = self.cfg.ObsCfg.num_observations
        hidden_dims = self.cfg.PolicyCfg.hidden_dims
        action_dim = self.cfg.ActionCfg.num_actions
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, hidden_dims[0]),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dims[1], hidden_dims[2]),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dims[2], action_dim),
        )
        self.to(self.device)

        self.checkpoint_path: Path | None = None

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)

    def load_policy(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {path}")

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        if "actor_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint does not contain actor_state_dict: {path}"
            )

        actor_state = checkpoint["actor_state_dict"]
        
        mlp_state = {
            key: value
            for key, value in actor_state.items()
            if key.startswith("mlp.")
        }
        

        self.load_state_dict(mlp_state, strict=True)
        self.eval().requires_grad_(False)
        self.checkpoint_path = path

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        
        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )
        observation_dim = self.cfg.ObsCfg.num_observations
        if tuple(observation.shape) == (observation_dim,):
            observation = observation.unsqueeze(0)

        with torch.inference_mode():
            action = self(observation)

        action = action.squeeze(0)
        
        return action
