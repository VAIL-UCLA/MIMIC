import os
import argparse
import glob
import numpy as np
import cv2
from PIL import Image
import imageio


def extract_pedestrians(video_path, output_dir, yolo_model="yolov8n-seg.pt",
                        conf=0.30, imgsz=1024):
    """Detect and segment all pedestrians (COCO class 0 = person) per frame."""
    from ultralytics import YOLO

    model = YOLO(yolo_model)
    results = model(source=video_path, stream=True, conf=conf, imgsz=imgsz)

    os.makedirs(output_dir, exist_ok=True)
    all_masks = []
    for i, result in enumerate(results):
        H, W = result.orig_shape
        mask = np.zeros((H, W), dtype=np.uint8)

        if result.masks is not None and result.boxes is not None:
            # COCO classes: 0=person, 1=bicycle, 2=car, 3=motorcycle,
            # 5=bus, 7=truck
            foreground_classes = {0, 1, 2, 3, 5, 7}
            for seg_mask, cls in zip(result.masks.data, result.boxes.cls):
                if int(cls.item()) not in foreground_classes:
                    continue
                raw = seg_mask.float().cpu().numpy()
                if raw.shape != (H, W):
                    raw = cv2.resize(raw, (W, H),
                                     interpolation=cv2.INTER_LINEAR)
                person_mask = (raw > 0.5).astype(np.uint8) * 255
                mask = np.maximum(mask, person_mask)

        # Morphological close to fill small holes, then smooth edges
        if mask.max() > 0:
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

        Image.fromarray(mask, mode="L").save(
            os.path.join(output_dir, f"{str(i).zfill(5)}.png")
        )
        all_masks.append(mask)

    # Save visualization video (white=pedestrian on black background)
    if all_masks:
        fps = 10

        vis_path = os.path.join(output_dir, "mask_vis.mp4")
        mask_rgb = [np.stack([m, m, m], axis=-1) for m in all_masks]
        imageio.mimwrite(vis_path, mask_rgb, fps=fps)
        print(f"  Mask video → {vis_path}")

        # Also save overlay: original with mask tinted
        overlay_path = os.path.join(output_dir, "overlay_vis.mp4")
        reader = imageio.get_reader(video_path)
        overlay_frames = []
        for idx, frame in enumerate(reader):
            if idx >= len(all_masks):
                break
            m = all_masks[idx].astype(np.float32) / 255.0
            tint = np.zeros_like(frame, dtype=np.float32)
            tint[:, :, 1] = 255  # green tint for pedestrians
            blended = frame.astype(np.float32) * (1 - m[..., None] * 0.5) \
                      + tint * (m[..., None] * 0.5)
            overlay_frames.append(np.clip(blended, 0, 255).astype(np.uint8))
        reader.close()
        imageio.mimwrite(overlay_path, overlay_frames, fps=fps)
        print(f"  Overlay video → {overlay_path}")

    print(f"Saved {len(all_masks)} pedestrian masks to {output_dir}")


def extract_foreground_sam2(video_path, output_dir, x, y,
                            model_name="sam2_b.pt", imgsz=1024):
    """SAM2 point-prompt based foreground extraction."""
    from ultralytics.models.sam import SAM2VideoPredictor

    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=imgsz, model=model_name)
    predictor = SAM2VideoPredictor(overrides=overrides)

    results = predictor(source=video_path, points=[x, y], labels=[1])

    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(results):
        mask_data = result.masks.data
        mask = mask_data[0].float().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask, mode="L").save(
            os.path.join(output_dir, f"{str(i).zfill(5)}.png")
        )

    print(f"Saved {len(results)} masks to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract foreground masks from video.")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file (.mp4), or directory with --batch.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save masks. Defaults to <video_dir>/masks/<video_stem>/.")
    parser.add_argument("--mode", type=str, default="pedestrian", choices=["pedestrian", "sam2"],
                        help="'pedestrian': auto-detect all persons via YOLO-seg. "
                             "'sam2': point-prompt via SAM2VideoPredictor.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (pedestrian mode).")
    parser.add_argument("--yolo_model", type=str, default="yolov8x-seg.pt",
                        help="YOLO segmentation model (pedestrian mode).")
    parser.add_argument("--x", type=int, default=960,
                        help="X coordinate of point prompt (sam2 mode).")
    parser.add_argument("--y", type=int, default=540,
                        help="Y coordinate of point prompt (sam2 mode).")
    parser.add_argument("--sam2_model", type=str, default="sam2_b.pt",
                        help="SAM2 model variant (sam2 mode).")
    parser.add_argument("--batch", action="store_true",
                        help="Treat --video_path as a directory and process all .mp4 files.")
    args = parser.parse_args()

    def get_output_dir(video_file):
        if args.output_dir:
            stem = os.path.splitext(os.path.basename(video_file))[0]
            return os.path.join(args.output_dir, stem)
        video_dir = os.path.dirname(video_file)
        stem = os.path.splitext(os.path.basename(video_file))[0]
        return os.path.join(video_dir, "masks", stem)

    def process_one(video_file):
        out_dir = get_output_dir(video_file)
        if args.mode == "pedestrian":
            extract_pedestrians(video_file, out_dir,
                                yolo_model=args.yolo_model, conf=args.conf)
        else:
            extract_foreground_sam2(video_file, out_dir,
                                   args.x, args.y, model_name=args.sam2_model)

    if args.batch:
        video_files = sorted(glob.glob(os.path.join(args.video_path, "*.mp4")))
        if not video_files:
            print(f"No .mp4 files found in {args.video_path}")
            return
        print(f"Found {len(video_files)} videos to process.")
        for vf in video_files:
            print(f"\nProcessing: {vf}")
            process_one(vf)
    else:
        process_one(args.video_path)


if __name__ == "__main__":
    main()
