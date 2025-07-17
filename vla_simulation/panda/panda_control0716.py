import cv2
import numpy as np
import os
from PIL import Image
from queue import Queue
import threading
import time
import traceback

import mujoco
import mujoco.viewer
import torch
import torch.cuda
from transformers import AutoProcessor, AutoModelForVision2Seq

import datasets
import io
from matplotlib import pyplot as plt



#cuda 설정
def setup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
#mujoco 모델 로드
def load_mujoco_model(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model file not found: {xml_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        return model, data
    except Exception as e:
        raise RuntimeError(f"Error loading MuJoCo model: {e}")
    
#mujoco 렌더러 설정
def setup_mujoco_renderer(model, image_height, image_width):
    try:
        renderer = mujoco.Renderer(model, height=image_height, width=image_width)
        return renderer
    except Exception as e:
        raise RuntimeError(f"Error setting up MuJoCo renderer: {e}")
    
#mujoco 뷰어 설정
def setup_mujoco_viewer(model, data):
    try:
        viewer = mujoco.viewer.launch_passive(model, data)

        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10
        viewer.cam.distance = 3.5

        return viewer
    except Exception as e:
        print(f"Could not launch passive viewer: {e}. Proceeding without visualization.")
        return None
        
#vla 모델 로드
def load_vla_model(model_name, mujoco_model):
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "8Gib"},
            offload_folder="offload")
        model.eval()


        if hasattr(model, "half"):
            model = model.half()

        if not hasattr(model.config, "action_dim"):
            print(f"Warning: Model {model_name} does not have an action dimension. Use MuJoCo's n_actions instead.")
            model.config.action_dim = mujoco_model.nu
        print(f"Model expected action dimension (from config or MuJoCo model): {model.config.action_dim}")

        n_actions_env = mujoco_model.nu
        print(f"MuJoCo environment action dimension: {n_actions_env}")

        if model.config.action_dim != n_actions_env:
            print(f"Warning: Model action dimension ({model.config.action_dim}) does not match MuJoCo environment action dimension ({n_actions_env}).")

        return model, processor
    except Exception as e:
        raise RuntimeError(f"Error loading VLA model: {e}")
    
def process_image(renderer, data, camera_name):
    renderer.update_scene(data, camera=camera_name)

    image_obs_rgb = renderer.render()
    image_obs_rgb = cv2.cvtColor(image_obs_rgb, cv2.COLOR_BGR2RGB)
    image_obs_rgb = Image.fromarray(image_obs_rgb)

    return image_obs_rgb

def prepare_model_inputs(processor, image, text, device):
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77)
    
    inputs = {k : v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in inputs.items()}

    # 이미지 텐서 처리 및 디바이스 이동
    # 1. pixel_values가 있으면 이미지 텐서의 차원을 조정 (배치 차원 제거)
    # 2. 모든 입력을 지정된 디바이스(GPU/CPU)로 이동
    if "pixel_values" in inputs:
        pixel_values = inputs["pixel_values"]

        # 5차원 텐서인 경우 배치 차원(두번째 차원) 제거 
        # [batch, 1, channel, height, width] -> [batch, channel, height, width]
        if len(pixel_values.shape) == 5:
            pixel_values = pixel_values.squeeze(1)
        
        inputs["pixel_values"] = pixel_values

    # 모든 입력 텐서를 지정된 디바이스로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs

def run_vla_inference(model, inputs, action_token_length, result_queue):
    try:
        if "input_ids" not in inputs or "attention_mask" not in inputs or "pixel_values" not in inputs:
            raise ValueError("Missing required input keys in inputs dictionary.")
        
        # torch.cuda.amp.autocast()를 사용하여 자동 혼합 정밀도(Automatic Mixed Precision) 연산 활성화
        # 이는 GPU 메모리 사용량을 줄이고 연산 속도를 높이는 데 도움이 됨
        with torch.cuda.amp.autocast():
            # torch.no_grad()로 gradient 계산을 비활성화하여 추론 시 메모리 사용량 감소
            with torch.no_grad():
                # VLA 모델에 입력 데이터를 전달하여 추론 수행
                # - input_ids: 텍스트를 토큰화한 ID 시퀀스
                # - attention_mask: 패딩 토큰을 구분하기 위한 마스크
                # - pixel_values: 이미지를 모델 입력 형식으로 변환한 텐서
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    return_dict=True)
                
                # 모델 출력에서 action token에 해당하는 logits만 추출
                action_logits = outputs.logits[:, -action_token_length:, :]
                # softmax를 적용하여 각 토큰의 확률 분포 계산
                action_probs = torch.softmax(action_logits, dim=-1)
                # 가장 높은 확률을 가진 토큰 선택
                action_tokens = torch.argmax(action_probs, dim=-1)

                # 결과 큐에 성공 상태와 예측된 action token을 전달
                result_queue.put(("success", action_tokens))
    except Exception as e:
        print(f"Error during VLA inference: {e}")
        traceback.print_exc()
        result_queue.put(("error", str(e)))

def decode_action_tokens(model, tokens):
    """
    VLA 모델이 생성한 action token을 실제 로봇 제어에 사용할 수 있는 연속적인 action 값으로 디코딩하는 함수
    
    Args:
        model: VLA 모델 객체 (action dimension 정보를 가짐)
        tokens: 모델이 생성한 이산적인 action token 텐서
        
    Returns:
        numpy array: -1.0 ~ 1.0 사이로 정규화된 연속적인 action 값
        
    동작 과정:
    1. 모델의 action dimension과 token 개수(256개)를 가져옴
    2. 입력 tokens에서 action에 해당하는 부분만 추출
    3. 0~255 사이의 이산 token 값을 -1~1 사이의 연속값으로 변환
    4. action dimension이 맞지 않는 경우 크기 조정
       - 너무 크면 앞부분만 사용
       - 너무 작으면 나머지는 0으로 패딩
    5. 최종 action 값을 -1 ~ 1 사이로 클리핑하여 반환
    """
    action_dim = model.config.action_dim  # 모델의 action dimension
    num_action_bins = 256  # action token의 개수 (0~255)
    action_tokens = tokens[0, -action_dim:]  # batch의 첫번째 시퀀스에서 action 부분만 추출

    # token을 -1~1 사이의 연속값으로 변환
    action_values = (action_tokens.float() / (num_action_bins - 1.0)) * 2.0 - 1.0
    action_np = action_values.detach().cpu().numpy()

    # action dimension 크기 조정
    if action_np.shape[0] != action_dim:
        if action_np.shape[0] > action_dim:
            action_np = action_np[:action_dim]  # 앞부분만 사용
        else:
            padded_action = np.zeros(action_dim)  # 부족한 부분은 0으로 패딩
            padded_action[:action_np.shape[0]] = action_np
            action_np = padded_action

    return np.clip(action_np, -1.0, 1.0)  # -1 ~ 1 사이로 클리핑

def convert_vla_action_to_mujoco_action(vla_action, mujoco_data):
    """
    VLA의 7차원 액션을 MuJoCo의 15차원 액션으로 변환
    
    VLA action (7차원): [Δx, Δy, Δz, Roll, Pitch, Yaw, Grip]
    MuJoCo action (15차원): [joint_x, joint_y, joint_z, joint_roll, joint_pitch, joint_yaw, 
                              joint1, joint2, joint3, joint4, joint5, joint6, joint7, 
                              gripper_left, gripper_right]
    
    Args:
        vla_action: 7차원 VLA 액션 배열
        mujoco_data: MuJoCo 데이터 객체
        
    Returns:
        15차원 MuJoCo 액션 배열
    """
    mujoco_action = np.zeros(15)
    
    # 가상 관절들 (그리퍼 위치 제어용)
    mujoco_action[0] = vla_action[0]  # joint_x (x position)
    mujoco_action[1] = vla_action[1]  # joint_y (y position)
    mujoco_action[2] = vla_action[2]  # joint_z (z position)
    mujoco_action[3] = vla_action[3]  # joint_roll (roll)
    mujoco_action[4] = vla_action[4]  # joint_pitch (pitch)
    mujoco_action[5] = vla_action[5]  # joint_yaw (yaw)
    
    # Pinocchio를 사용한 역기구학 계산
    import pinocchio as pin
    from example_robot_data import load
    
    # 로봇 모델 로드
    # Pinocchio 라이브러리를 사용하여 Panda 로봇의 URDF 파일을 로드하고 RobotWrapper 객체를 생성
    # RobotWrapper는 로봇의 기구학적 모델과 동역학적 계산을 위한 인터페이스를 제공
    
    robot = load("panda")
    model = robot.model
    data = robot.data

    # 현재 엔드이펙터 위치/방향 계산
    # 현재 관절 각도를 MuJoCo 데이터에서 가져와야 함
    current_q = mujoco_data.qpos[6:13]  # joint1-7의 현재 각도 값

    pin.forwardKinematics(model, data, current_q)
    current_pose = data.oMi[7]  # 엔드이펙터 프레임
    breakpoint()
    # 목표 위치/방향 설정 (VLA action에서 변환)
    target_translation = np.array([vla_action[0], vla_action[1], vla_action[2]])
    target_rotation = pin.rpy.rpyToMatrix(vla_action[3], vla_action[4], vla_action[5])
    target_pose = pin.SE3(target_rotation, target_translation)
    
    # 역기구학 계산
    eps = 1e-4
    max_iter = 100
    damp = 1e-12
    q = pin.neutral(model)
    
    success = False
    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        dMi = target_pose.actInv(data.oMi[7])
        err = pin.log(dMi).vector
        if np.linalg.norm(err) < eps:
            success = True
            break
        J = pin.computeJointJacobian(model, data, q)
        q = q + np.linalg.solve(J.T@J + damp * np.eye(7), J.T@err)
    
    if success:
        # 계산된 관절 각도를 MuJoCo action으로 변환
        mujoco_action[6:13] = q  # joint1-7
    else:
        # 역기구학 실패시 기본값 사용
        mujoco_action[6:13] = np.zeros(7)
    
    # 그리퍼 제어
    gripper_value = vla_action[6]  # VLA의 그리퍼 값
    mujoco_action[13] = gripper_value  # gripper_left
    mujoco_action[14] = gripper_value  # gripper_right
    
    return mujoco_action

def compute_default_action(data, target_pos):
    # 이 함수는 VLA inference를 사용하지 않고 단순히 기본적인 제어 로직을 구현한 것입니다.
    # VLA inference는 run_vla_inference() 함수에서 수행됩니다.
    current_pos = data.site_xpos[0]
    direction = target_pos - current_pos
    direction = direction / (np.linalg.norm(direction) + 1e-6)

    # 9차원 액션으로 변경 (Panda 로봇 7자유도 + 그리퍼 2자유도)
    predicted_action = np.zeros(9)

    # 7개의 관절을 모두 사용하여 방향 제어
    # 각 관절의 움직임을 조절하여 더 자연스러운 동작 생성
    predicted_action[0] = direction[0] * 0.3  # base rotation
    predicted_action[1] = direction[1] * 0.3  # shoulder
    predicted_action[2] = direction[2] * 0.3  # elbow
    predicted_action[3] = direction[0] * 0.2  # wrist1 
    predicted_action[4] = direction[1] * 0.2  # wrist2
    predicted_action[5] = direction[2] * 0.2  # wrist3
    predicted_action[6] = direction[0] * 0.1  # hand rotation
    
    # 그리퍼 제어 (8,9번 인덱스)
    predicted_action[7] = 0.0  # left finger
    predicted_action[8] = 0.0  # right finger

    return predicted_action




if __name__ == "__main__":
    MUJOCO_MODEL_PATH = "assets/7DOF_panda_0716.xml"
    OPENVLA_MODEL_NAME = "openvla/openvla-7b"
    RENDER_CAMERA_NAME = "robotview"  # 로봇이 바라보는 시점의 카메라 뷰

    #LANGUAGE_INSTRUCTION = "Pick the red ball"

    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256

    SIM_DURATION_SECONDS = 20
    CONTROL_FREQ = 10
    RENDER_FREQ = 30

    print("Initializing Panda Robot Simulation...")
    device = setup_cuda()
    print(f"Using device: {device}")

    model, data = load_mujoco_model(MUJOCO_MODEL_PATH)
    renderer = setup_mujoco_renderer(model, IMAGE_HEIGHT, IMAGE_WIDTH)
    # 시뮬레이션 화면을 실시간으로 보여주는 뷰어 생성 - 여기서 디스플레이 창이 생성됨
    # 뷰어는 로봇의 움직임과 환경을 시각화하여 모니터링할 수 있게 해줌
    # viewer = setup_mujoco_viewer(model, data)  # GLFW 오류 방지를 위해 주석 처리
    viewer = None

    control_timestep = 1.0 / CONTROL_FREQ
    render_timestep = 1.0 / RENDER_FREQ

    print(f"Simulation timestep: {model.opt.timestep: .4f} seconds")  # 물리 시뮬레이션의 기본 시간 간격 (물리 법칙이 적용되는 간격)
    print(f"Control timestep: {control_timestep: .4f} seconds")       # 로봇 제어 명령이 전송되는 시간 간격 (로봇 동작 제어 주기)
    print(f"Render timestep: {render_timestep: .4f} seconds")         # 화면이 갱신되는 시간 간격 (시각화 업데이트 주기)

    # 시뮬레이션 시간 및 카운터 초기화
    sim_time = 0.0                      # 시뮬레이션 진행 시간 (초)
    control_step_counter = 0            # 로봇 제어 스텝 카운터
    render_step_counter = 0             # 화면 렌더링 스텝 카운터
    last_control_time = time.time()     # 마지막 로봇 제어 시간
    last_render_time = time.time()      # 마지막 화면 렌더링 시간
        
    ##----------------------------------------------------------------------------------##
    ####-demo에서 추가-######

    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # 동일한 데이터셋 가져오기
    ds = datasets.load_dataset("jxu124/OpenX-Embodiment", "nyu_door_opening_surprising_effectiveness", trust_remote_code=True)#, streaming=True, split='train')  # IterDataset
    samples = ds['train'][1]['data.pickle']['steps'] # 25 steps, action/observation 등

    images = []
    actions = []
    frame_indices = []
    frame_idx = 0
    ground_truth_actions = []
    
    # 수정
    for k, sample in enumerate(samples):
        #이미지
        encoded_image = sample['observation']['image']['bytes'] # 
        #instruction
        bytes_str = sample['observation']['natural_language_instruction'] # 
        INSTRUCTION = bytes_str.decode("utf-8")  # UTF-8 디코딩
        # 바이트 데이터를 이미지로 변환
        image = Image.open(io.BytesIO(encoded_image))
        images.append(image)

        # 실제 로봇 action
        # 데이터셋에서 실제 로봇이 수행한 액션값을 추출 (ground truth)
        # 이 값은 VLA 모델의 action 예측값과 비교하기 위한 참조용으로,
        # 실제 predict 함수에서는 사용되지 않음
        # VLA는 이미지와 instruction만을 입력으로 받아 독립적으로 액션을 예측함
        ground_truth_action = sample['action']['world_vector'] + sample['action']['rotation_delta'] + sample['action']['gripper_closedness_action']
        ground_truth_actions.append(ground_truth_action)
        prompt = f"In: What action should the robot take to {INSTRUCTION} Out:"

        # Predict Action (7-DoF; un-normalize for BridgeV2)
        # 로봇의 7자유도(7-DoF) 동작을 예측하기 위한 입력 처리
        # - Δx, Δy, Δz: 로봇 엔드이펙터의 3차원 이동 방향과 크기
        # - Roll, Pitch, Yaw: 로봇 엔드이펙터의 3차원 회전 방향
        # - Grip: 그리퍼의 열고 닫힘 정도
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        
        #  예측(inference)
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        actions.append(action)

        frame_indices.append(frame_idx)
        frame_idx += 1


    actions = np.array(actions)
    ground_truth_actions = np.array(ground_truth_actions)
    ####-demo에서 추가-######
    ##----------------------------------------------------------------------------------##

    print("Starting Panda robot simulation...")
    try:
        # model과 data는 load_mujoco_model()에서 로드한 MuJoCo 모델과 데이터입니다.
        # model은 assets/robot_gripper_with_target.xml 파일에서 로드된 로봇 모델이고
        # data는 해당 모델의 상태를 저장하는 데이터 구조입니다.
        mujoco.mj_resetData(model, data)
        start_time = time.time()
        
        # Ground truth actions 인덱스 초기화
        action_index = 0
        total_ground_truth_actions = len(ground_truth_actions)
        print(f"Total ground truth actions available: {total_ground_truth_actions}")

        while sim_time < SIM_DURATION_SECONDS:
            current_loop_time = time.time()

            if sim_time == 0 or current_loop_time - last_control_time > control_timestep:
                last_control_time = current_loop_time

                # Ground truth action 사용
                if action_index < total_ground_truth_actions:
                    # VLA의 7차원 액션을 MuJoCo의 15차원 액션으로 변환
                    vla_action = ground_truth_actions[action_index]
                    predicted_action = convert_vla_action_to_mujoco_action(vla_action, data)
                    
                    print(f"Step {action_index + 1}/{total_ground_truth_actions}")
                    print(f"VLA action (7D): {vla_action}")
                    print(f"MuJoCo action (15D): {predicted_action}")
                    
                    action_index += 1
                else:
                    # 모든 ground truth actions를 사용한 후 기본 액션 사용
                    print("All ground truth actions used, using default action")
                    predicted_action = compute_default_action(data, np.array([0.3, 0.1, 0.5])) 

                # VLA 모델이 예측한 액션값을 -1.0 ~ 1.0 사이로 제한하여 안전한 제어 범위 내로 조정
                predicted_action = np.clip(predicted_action, -1.0, 1.0)

                # Print the sequence of predicted actions
                print("Action sequence:")
                action_labels = ["joint_x", "joint_y", "joint_z", "joint_roll", "joint_pitch", "joint_yaw",
                               "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", 
                               "gripper_left", "gripper_right"]
                for i, (action_value, label) in enumerate(zip(predicted_action, action_labels)):
                    print(f"{label}: {action_value:.4f}")
                
                # 제한된 액션값을 로봇의 제어 입력으로 설정
                data.ctrl[:] = predicted_action
                
                # 로봇 제어 횟수 카운터 증가
                control_step_counter += 1

                
            # mj_step()은 MuJoCo 시뮬레이션을 한 타임스텝 진행시키는 함수입니다.
            # model: 로봇의 물리적 특성, 관절 구조 등이 정의된 MuJoCo 모델
            # data: 로봇의 현재 상태(관절 각도, 속도 등)를 담고 있는 데이터
            # 이 함수는 현재 제어 입력(data.ctrl)에 따라 로봇의 다음 상태를 계산합니다
            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep

            # 뷰어가 실행 중이고 렌더링 시간이 지났으면 화면 업데이트
            if viewer and viewer.is_running():
                if current_loop_time - last_render_time > render_timestep:
                    # 뷰어 화면 동기화 
                    viewer.sync()
                    
                    # 마지막 렌더링 시간과 카운터 업데이트
                    last_render_time = current_loop_time  
                    render_step_counter += 1
            # 뷰어가 종료되었으면 시뮬레이션도 종료
            elif viewer and not viewer.is_running():
                break
    # 이 코드는 시뮬레이션 실행 중 발생할 수 있는 예외 상황을 처리하는 예외 처리 블록입니다:

    # 1. KeyboardInterrupt 예외 처리: 
    # 사용자가 Ctrl+C를 눌러 시뮬레이션을 중단했을 때 실행됨
    except KeyboardInterrupt:
        print("Simulation interrupted by user. Exiting...")

    # 2. 기타 모든 예외 처리:
    # 시뮬레이션 중 발생하는 다른 모든 에러를 잡아서 출력
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()  # 상세한 에러 추적 정보 출력

    #그냥
    # 3. finally 블록:
    # 예외 발생 여부와 관계없이 항상 실행되는 정리 코드
    # - 뷰어가 실행 중이면 종료
    # - 시뮬레이션 종료 시간 기록 
    finally:
        if viewer and viewer.is_running():
            viewer.close()
        
        end_time = time.time()

        print("Panda robot simulation complete.")
        print(f"Total simulation time: {sim_time: .2f} seconds")
        print(f"Total real time: {end_time - start_time: .2f} seconds")
        print(f"Control steps: {control_step_counter}")
        print(f"Render steps: {render_step_counter}")