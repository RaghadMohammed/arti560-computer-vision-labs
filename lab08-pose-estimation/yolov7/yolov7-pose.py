# FILE: yolov7/yolov7-pose.py

from pathlib import Path
import time
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

FILE = Path(__file__).resolve()
ROOT = FILE.parent
PROJECT_ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


INPUT_SIZE = 256
CONF_THRES = 0.25
IOU_THRES = 0.65
NUM_CLASSES = 1
NUM_KEYPOINTS = 17

VIDEO_NAME = "yoga"  
VIDEO_PATH = PROJECT_ROOT / "media" / f"{VIDEO_NAME}.mp4"
WEIGHTS_PATH = ROOT / "yolov7-w6-pose.pt"
OUTPUT_PATH = PROJECT_ROOT / f"{VIDEO_NAME}_yolov7.avi"


if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

print("Selected Device:", DEVICE)


def load_model(weights_path: Path) -> torch.nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    weights = torch.load(
        str(weights_path),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model = weights["model"]
    model = model.float().eval().to(DEVICE)
    return model


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    image = letterbox(frame, INPUT_SIZE, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]), dtype=torch.float32)
    image = image.to(DEVICE)
    return image


def run_pose_inference(model: torch.nn.Module, frame: np.ndarray) -> tuple[np.ndarray, float]:
    image = preprocess_frame(frame)

    with torch.no_grad():
        start = time.time()
        output, _ = model(image)
        end = time.time()

        fps = 1.0 / max(end - start, 1e-6)
        output = non_max_suppression_kpt(
            output,
            CONF_THRES,
            IOU_THRES,
            nc=NUM_CLASSES,
            nkpt=NUM_KEYPOINTS,
            kpt_label=True,
        )
        output = output_to_keypoint(output)

    rendered = image[0].permute(1, 2, 0) * 255
    rendered = rendered.cpu().numpy().astype(np.uint8)
    rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(rendered, output[idx, 7:].T, 3)

    cv2.putText(
        rendered,
        f"FPS : {fps:.2f}",
        (140, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        rendered,
        "YOLOv7 Pose",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return rendered, fps


def create_video_writer(model: torch.nn.Module, video_path: Path, output_path: Path) -> tuple[cv2.VideoCapture, cv2.VideoWriter]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError(f"Could not read first frame from: {video_path}")

    processed_frame, _ = run_pose_inference(model, first_frame)
    height, width, _ = processed_frame.shape

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, writer


def main() -> None:
    model = load_model(WEIGHTS_PATH)
    cap, writer = create_video_writer(model, VIDEO_PATH, OUTPUT_PATH)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nUnable to read frame. Exiting ..")
            break

        rendered_frame, fps = run_pose_inference(model, frame)
        writer.write(rendered_frame)
        cv2.imshow("YOLOv7 Pose Output", rendered_frame)

        frame_count += 1
        print(f"Processed frame {frame_count} | FPS {fps:.2f}", end="\r")

        if cv2.waitKey(1) == ord("q"):
            print("\nStopped by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nYOLOv7 Pose completed successfully.")
    print(f"Input : {VIDEO_PATH}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()