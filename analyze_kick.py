"""
analyze_kick.py

Usage:
    python analyze_kick.py path/to/your_kick_video.mp4 [--side right|left] [--out output_dir]

What it does (both validation tracks in one pass):

  TRACK 1 - Feedback accuracy:
    Finds the peak-kick frame, computes hip rotation / standing leg / torso lean /
    guard position, and prints plain-language coaching feedback. Show this output
    to training partners and ask: "does this feel accurate?"

  TRACK 2 - Motion blur / frame-rate reliability:
    Reports MediaPipe's per-frame landmark visibility (confidence) across the
    whole clip, and flags what fraction of frames near the kick had reliable
    tracking. Run the same test on a normal-speed clip and a slo-mo clip of the
    same kick and compare the numbers directly.

Outputs:
  - Console report (both tracks)
  - <out>/annotated.mp4        - video with skeleton overlay + peak frame marked
  - <out>/confidence_report.csv - per-frame visibility scores for Track 2 analysis
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp

from pose_extract import (
    extract_frames,
    compute_kick_metrics,
    find_peak_kick_frame,
    generate_feedback,
    generate_experimental_feedback,
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def run_track2_confidence_report(all_frames, kicking_side, peak_idx, out_dir, window=10, verbose=True):
    """
    Writes a per-frame CSV of key-landmark visibility scores (always), and
    prints a summary of tracking reliability around the peak-kick frame when
    verbose=True. CSV writing is unconditional -- both the CLI and the web
    app need the file on disk, only the console narration differs.
    """
    side = kicking_side.upper()
    key_landmarks = [f"{side}_ANKLE", f"{side}_KNEE", f"{side}_HIP", f"{side}_WRIST"]

    csv_path = os.path.join(out_dir, "confidence_report.csv")
    rows = []
    for f in all_frames:
        row = {"frame_idx": f["frame_idx"]}
        if f["landmarks"] is None:
            for k in key_landmarks:
                row[k] = ""
        else:
            for k in key_landmarks:
                row[k] = round(f["landmarks"][k][2], 3) if k in f["landmarks"] else ""
        rows.append(row)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame_idx"] + key_landmarks)
        writer.writeheader()
        writer.writerows(rows)

    # Reliability window around the peak kick frame
    reliable_count = 0
    total_count = 0
    if peak_idx is not None:
        lo, hi = max(0, peak_idx - window), peak_idx + window
        for f in all_frames:
            if lo <= f["frame_idx"] <= hi:
                total_count += 1
                if f["landmarks"] is not None:
                    vis_scores = [f["landmarks"][k][2] for k in key_landmarks if k in f["landmarks"]]
                    if vis_scores and min(vis_scores) >= 0.5:
                        reliable_count += 1

    pct_reliable = (reliable_count / total_count * 100) if total_count else 0.0

    if verbose:
        print("\n=== TRACK 2: Tracking reliability around the kick ===")
        if peak_idx is None:
            print("Could not identify a clear peak-kick frame -- tracking was too unreliable "
                  "throughout the clip to even locate the kick. This itself is a signal: "
                  "this footage/frame-rate combination is not usable as-is.")
        else:
            print(f"Peak-kick frame: {peak_idx}")
            print(f"Frames within +/-{window} of peak with reliable landmark visibility "
                  f"(>=0.5 confidence): {reliable_count}/{total_count} ({pct_reliable:.0f}%)")
            if pct_reliable >= 80:
                print("-> Reliable. This frame rate / footage quality is usable.")
            elif pct_reliable >= 50:
                print("-> Marginal. Some frames near the kick are unreliable -- consider "
                      "slo-mo capture or better lighting/framing.")
            else:
                print("-> Unreliable. Motion blur is likely too severe at this frame rate. "
                      "Try filming in slo-mo (120fps+) instead.")
        print(f"Full per-frame data written to: {csv_path}")

    return pct_reliable


def write_annotated_video(video_path, all_frames, peak_idx, out_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(out_dir, "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 30, (w, h))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fdata = all_frames[idx] if idx < len(all_frames) else None
        if fdata and fdata["raw_result"] is not None:
            mp_drawing.draw_landmarks(
                frame,
                fdata["raw_result"],
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        if idx == peak_idx:
            cv2.putText(frame, "PEAK KICK FRAME", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"\nAnnotated video written to: {out_path}")


def run_analysis(video_path, side, out_dir, verbose=True):
    """
    The single entry point both the CLI and the web app call. Runs the full
    pipeline on one video and returns a structured result dict instead of just
    printing to console, so a web app can render it.

    Returns:
        {
          "fps": float,
          "frame_count": int,
          "peak_idx": int | None,
          "feedback": [(label, status, message), ...] | None,
          "metrics": {metric_name: value} | None,
          "confidence_pct": float,
          "annotated_video_path": str,
          "confidence_csv_path": str,
        }
    """
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"Processing {video_path} ...")
    all_frames, fps = extract_frames(video_path)
    if verbose:
        print(f"Extracted {len(all_frames)} frames at {fps:.1f} fps.")

    peak_idx = find_peak_kick_frame(all_frames, kicking_side=side)

    metrics, feedback = None, None
    if peak_idx is not None:
        peak_frame = next(f for f in all_frames if f["frame_idx"] == peak_idx)
        metrics = compute_kick_metrics(peak_frame["landmarks"], kicking_side=side)
        feedback = generate_feedback(metrics)
        if verbose:
            print("\n=== TRACK 1: Coaching feedback ===")
            print(f"Analyzed at frame {peak_idx} (peak of kick):\n")
            for label, status, message in feedback:
                marker = {"ok": "[OK]", "low": "[!]", "high": "[!]", "unmeasured": "[?]"}[status]
                print(f"  {marker} {label}: {message}")
    elif verbose:
        print("\n=== TRACK 1: Coaching feedback ===")
        print("Could not find a reliable peak-kick frame to analyze. "
              "See Track 2 results below -- this is likely a tracking issue, not a technique issue.")

    confidence_pct = run_track2_confidence_report(all_frames, side, peak_idx, out_dir, verbose=verbose)

    # Experimental metrics -- kept structurally separate from the validated
    # `feedback` above. Currently just foot pivot. Not blended with the core
    # feedback list on purpose: it hasn't earned the same confidence yet.
    experimental_feedback, pivot_debug = generate_experimental_feedback(all_frames, kicking_side=side)
    if verbose:
        print("\n=== EXPERIMENTAL: foot pivot (not yet fully validated) ===")
        for label, status, message in experimental_feedback:
            marker = {"ok": "[OK]", "low": "[!]", "high": "[!]", "unmeasured": "[?]"}[status]
            print(f"  {marker} {label}: {message}")

    write_annotated_video(video_path, all_frames, peak_idx, out_dir)

    return {
        "fps": fps,
        "frame_count": len(all_frames),
        "peak_idx": peak_idx,
        "feedback": feedback,
        "metrics": metrics,
        "confidence_pct": confidence_pct,
        "experimental_feedback": experimental_feedback,
        "annotated_video_path": os.path.join(out_dir, "annotated.mp4"),
        "confidence_csv_path": os.path.join(out_dir, "confidence_report.csv"),
    }


def main():
    parser = argparse.ArgumentParser(description="Roundhouse kick pose analyzer (validation prototype)")
    parser.add_argument("video", help="Path to the input video clip")
    parser.add_argument("--side", choices=["right", "left"], default="right",
                         help="Which leg is kicking (default: right)")
    parser.add_argument("--out", default="./output", help="Output directory")
    args = parser.parse_args()

    run_analysis(args.video, args.side, args.out, verbose=True)


if __name__ == "__main__":
    main()
