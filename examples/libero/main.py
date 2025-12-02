import collections
import dataclasses
import logging
import math
import os
import pathlib
from enum import Enum

import imageio
import numpy as np
import tqdm
import tyro
import yaml

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from libero_pro import perturbation  # NOTE: adjust to "LIBERO-PRO" package name if needed


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class TaskSuite(str, Enum):
    # Vanilla LIBERO
    LIBERO_GOAL = "libero_goal"
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_10 = "libero_10"
    LIBERO_OBJECT = "libero_object"
    LIBERO_90 = "libero_90"

    # LIBERO-PRO variants
    LIBERO_GOAL_TEMP = "libero_goal_temp"
    LIBERO_SPATIAL_TEMP = "libero_spatial_temp"
    LIBERO_10_TEMP = "libero_10_temp"
    LIBERO_OBJECT_TEMP = "libero_object_temp"

    LIBERO_GOAL_LAN = "libero_goal_lan"
    LIBERO_SPATIAL_LAN = "libero_spatial_lan"
    LIBERO_10_LAN = "libero_10_lan"
    LIBERO_OBJECT_LAN = "libero_object_lan"

    LIBERO_GOAL_OBJECT = "libero_goal_object"
    LIBERO_SPATIAL_OBJECT = "libero_spatial_object"
    LIBERO_10_OBJECT = "libero_10_object"
    LIBERO_OBJECT_OBJECT = "libero_object_object"

    LIBERO_GOAL_SWAP = "libero_goal_swap"
    LIBERO_SPATIAL_SWAP = "libero_spatial_swap"
    LIBERO_10_SWAP = "libero_10_swap"
    LIBERO_OBJECT_SWAP = "libero_object_swap"

    LIBERO_GOAL_TASK = "libero_goal_task"
    LIBERO_SPATIAL_TASK = "libero_spatial_task"
    LIBERO_10_TASK = "libero_10_task"
    LIBERO_OBJECT_TASK = "libero_object_task"

    LIBERO_GOAL_ENV = "libero_goal_env"
    LIBERO_SPATIAL_ENV = "libero_spatial_env"
    LIBERO_10_ENV = "libero_10_env"
    LIBERO_OBJECT_ENV = "libero_object_env"


TASK_MAX_STEPS = {
    # Vanilla suites
    "libero_spatial": 220,  # longest training demo ~193
    "libero_object": 280,   # longest training demo ~254
    "libero_goal": 300,     # longest training demo ~270
    "libero_10": 520,       # longest training demo ~505
    "libero_90": 400,       # longest training demo ~373

    # LIBERO-PRO suites (match base suite step limits)
    "libero_goal_temp": 300,
    "libero_spatial_temp": 220,
    "libero_10_temp": 520,
    "libero_object_temp": 280,

    "libero_goal_lan": 300,
    "libero_spatial_lan": 220,
    "libero_10_lan": 520,
    "libero_object_lan": 280,

    "libero_goal_object": 300,
    "libero_spatial_object": 220,
    "libero_10_object": 520,
    "libero_object_object": 280,

    "libero_goal_swap": 300,
    "libero_spatial_swap": 220,
    "libero_10_swap": 520,
    "libero_object_swap": 280,

    "libero_goal_task": 300,
    "libero_spatial_task": 220,
    "libero_10_task": 520,
    "libero_object_task": 280,

    "libero_goal_env": 300,
    "libero_spatial_env": 220,
    "libero_10_env": 520,
    "libero_object_env": 280,
}


@dataclasses.dataclass
class Args:
    ###########################################################################
    # Model server parameters
    ###########################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    ###########################################################################
    # LIBERO environment-specific parameters
    ###########################################################################
    # You can pass any of the TaskSuite values here, e.g. libero_goal_env
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    ###########################################################################
    # LIBERO-PRO evaluation config
    ###########################################################################
    # Path to the LIBERO-PRO evaluation_config.yaml
    evaluation_config_path: str = "evaluation_config.yaml"

    ###########################################################################
    # Utils
    ###########################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random seed (for reproducibility)


def prepare_libero_pro_env(args: Args) -> None:
    """
    Prepare perturbed BDDL / init files for LIBERO-PRO and possibly rewrite
    args.task_suite_name to include a perturbation suffix or *_temp.

    This mirrors the logic from the LIBERO-PRO repo's eval_libero / perturbation code.
    """
    if not os.path.exists(args.evaluation_config_path):
        logging.info("No evaluation_config.yaml found; running vanilla LIBERO.")
        return

    with open(args.evaluation_config_path, "r", encoding="utf-8") as f:
        evaluation_cfg = yaml.safe_load(f)

    # Base paths get the suite name appended
    evaluation_cfg["bddl_files_path"] = evaluation_cfg.get("bddl_files_path", "") + "/" + args.task_suite_name
    evaluation_cfg["task_suite_name"] = args.task_suite_name

    use_swap = evaluation_cfg.get("use_swap", False)
    use_object = evaluation_cfg.get("use_object", False)
    use_language = evaluation_cfg.get("use_language", False)
    use_task = evaluation_cfg.get("use_task", False)
    use_environment = evaluation_cfg.get("use_environment", False)

    flags = [use_swap, use_object, use_language, use_task, use_environment]

    # Case 1: more than one flag True -> combined perturbations -> *_temp
    if sum(flags) > 1:
        bddl_file_path = evaluation_cfg.get("bddl_files_path", "") + args.task_suite_name + "_temp/"
        init_file_path = evaluation_cfg.get("init_file_dir", "") + args.task_suite_name + "_temp/"

        if not os.path.exists(bddl_file_path) or not os.path.exists(init_file_path):
            os.makedirs(init_file_path, exist_ok=True)
            os.makedirs(bddl_file_path, exist_ok=True)

            log_content = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
            with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                log_file.write(log_content)

            perturbation.create_env(configs=evaluation_cfg)
        else:
            with open(os.path.join(bddl_file_path, "log.txt"), "r") as log_file:
                log_contents = log_file.read().strip()

            expected_log = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"

            if log_contents != expected_log:
                # Wipe existing dirs
                for folder in [bddl_file_path, init_file_path]:
                    for root, dirs, files in os.walk(folder, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))

                os.makedirs(init_file_path, exist_ok=True)
                os.makedirs(bddl_file_path, exist_ok=True)

                with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                    log_file.write(expected_log)

                perturbation.create_env(configs=evaluation_cfg)

        # Update suite name to *_temp so LIBERO uses the right BDDL set
        args.task_suite_name = args.task_suite_name + "_temp"
        return

    # Case 2: exactly one perturbation flag True -> *_lan, *_swap, *_object, *_task, *_env
    if sum(flags) == 1:
        if use_swap:
            perturb_key = "use_swap"
        elif use_object:
            perturb_key = "use_object"
        elif use_language:
            perturb_key = "use_language"
        elif use_task:
            perturb_key = "use_task"
        elif use_environment:
            perturb_key = "use_environment"
        else:
            return  # should be unreachable, but safe-guard

        mapping = evaluation_cfg.get("perturbation_mapping", {})
        suffix = mapping.get(perturb_key, "")
        init_file_path = evaluation_cfg.get("init_file_dir", "") + args.task_suite_name + "_" + suffix

        if not os.path.exists(init_file_path):
            perturbation.create_env(configs=evaluation_cfg)

        args.task_suite_name = args.task_suite_name + "_" + suffix
        return

    # Case 3: no perturbation flags True -> vanilla LIBERO, do nothing
    logging.info("No LIBERO-PRO perturbation flags set; using vanilla task suite.")
    return


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Prepare perturbed LIBERO-PRO environment if config is present
    prepare_libero_pro_env(args)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Use unified max-steps table
    if args.task_suite_name in TASK_MAX_STEPS:
        max_steps = TASK_MAX_STEPS[args.task_suite_name]
    else:
        raise ValueError(f"Unknown task suite for max steps: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # Do nothing for the first few timesteps to let objects settle
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image (rotate 180° to match training preprocessing)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        element = {
                            "query_top_image": img,
                            "query_right_image": img,
                            "query_wrist_image": wrist_img,
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log per-task results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    # Seed affects object positions even with fixed init state
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490-L512
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
