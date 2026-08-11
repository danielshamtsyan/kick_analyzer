"""
pose_extract.py

Core pose-extraction and angle-computation logic for the roundhouse kick analyzer.
Uses MediaPipe's legacy Pose solution (bundled model, no network download needed).

This module does two jobs, matching the two validation tracks:
  1. TRACK 1 (feedback accuracy): extract joint angles at the peak-kick frame and
     compare them to reference ranges -> plain-language coaching cues.
  2. TRACK 2 (motion blur / frame-rate reliability): report per-frame landmark
     visibility/confidence so you can see whether normal-speed phone video is
     usable, or whether slo-mo capture is required.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Landmark indices we care about for a roundhouse kick (MediaPipe Pose, 33-point model)
LM = mp_pose.PoseLandmark


def angle_between(a, b, c):
    """Angle at point b, formed by points a-b-c, in degrees. a/b/c are (x, y) tuples."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def line_angle(p1, p2):
    """Angle of the line p1->p2 relative to horizontal, in degrees (0-180)."""
    p1, p2 = np.array(p1), np.array(p2)
    d = p2 - p1
    return float(np.degrees(np.arctan2(d[1], d[0])))


def extract_frames(video_path):
    """
    Runs MediaPipe Pose over every frame of the video.
    Returns a list of dicts, one per frame:
      { 'landmarks': {name: (x, y, visibility)}, 'frame_idx': int }
    Coordinates are normalized (0-1) image coordinates as MediaPipe returns them.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    results_per_frame = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # "full" model -- bundled offline, no network download needed.
                                      # (complexity=2 "heavy" requires downloading a model file at
                                      # runtime, which needs network access to storage.googleapis.com)
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            frame_data = {"frame_idx": frame_idx, "landmarks": None, "raw_result": None}
            if result.pose_landmarks:
                lm_dict = {}
                for name in LM:
                    p = result.pose_landmarks.landmark[name.value]
                    lm_dict[name.name] = (p.x, p.y, p.visibility)
                frame_data["landmarks"] = lm_dict
                frame_data["raw_result"] = result.pose_landmarks
            results_per_frame.append(frame_data)
            frame_idx += 1

    cap.release()
    return results_per_frame, fps


def compute_kick_metrics(landmarks, kicking_side="right"):
    """
    Given one frame's landmark dict, compute the metrics we score a roundhouse kick on.
    kicking_side: "right" or "left" -- which leg is kicking.
    Returns a dict of metric_name -> value (degrees), or None for metrics that
    can't be computed (missing/low-visibility landmarks).
    """
    side = kicking_side.upper()
    other = "LEFT" if side == "RIGHT" else "RIGHT"

    def pt(name):
        return landmarks[name][:2]

    def vis(name):
        return landmarks[name][2]

    metrics = {}

    # 1. Hip rotation proxy: angle of the hip line (left_hip -> right_hip) relative
    #    to horizontal. In a side-on kick, a well-rotated hip line tilts significantly
    #    away from horizontal compared to a square, un-rotated stance.
    try:
        hip_line_angle = abs(line_angle(pt(f"{other}_HIP"), pt(f"{side}_HIP")))
        metrics["hip_line_angle_deg"] = hip_line_angle
        metrics["hip_line_visibility"] = min(vis(f"{other}_HIP"), vis(f"{side}_HIP"))
    except KeyError:
        metrics["hip_line_angle_deg"] = None

    # 2. Standing (base) leg knee bend: hip-knee-ankle angle on the NON-kicking leg.
    #    Some bend (~150-165 deg, i.e. not locked straight) helps pivot power.
    try:
        standing_knee_angle = angle_between(
            pt(f"{other}_HIP"), pt(f"{other}_KNEE"), pt(f"{other}_ANKLE")
        )
        metrics["standing_knee_angle_deg"] = standing_knee_angle
        metrics["standing_knee_visibility"] = min(
            vis(f"{other}_HIP"), vis(f"{other}_KNEE"), vis(f"{other}_ANKLE")
        )
    except KeyError:
        metrics["standing_knee_angle_deg"] = None

    # 3. Torso lean: angle of the shoulder-to-hip line (kicking side) relative to
    #    vertical. Excessive backward lean bleeds power out of the kick.
    try:
        torso_angle = line_angle(pt(f"{side}_HIP"), pt(f"{side}_SHOULDER"))
        # Convert to lean-from-vertical (0 = perfectly upright)
        lean_from_vertical = abs(90 - abs(torso_angle))
        metrics["torso_lean_deg"] = lean_from_vertical
        metrics["torso_visibility"] = min(vis(f"{side}_HIP"), vis(f"{side}_SHOULDER"))
    except KeyError:
        metrics["torso_lean_deg"] = None

    # 4. Guard hand height: is the non-kicking-side hand still up near the head,
    #    or has it dropped? Measured as wrist height relative to shoulder height
    #    (normalized image y-coords, smaller y = higher up the frame).
    try:
        guard_wrist_y = pt(f"{other}_WRIST")[1]
        guard_shoulder_y = pt(f"{other}_SHOULDER")[1]
        # Positive = wrist is below shoulder (guard dropped); negative = wrist above shoulder
        guard_drop = guard_wrist_y - guard_shoulder_y
        metrics["guard_drop_normalized"] = guard_drop
        metrics["guard_visibility"] = min(vis(f"{other}_WRIST"), vis(f"{other}_SHOULDER"))
    except KeyError:
        metrics["guard_drop_normalized"] = None

    return metrics


def find_kick_window(all_frames, kicking_side="right", height_margin=0.08, min_window_frames=4):
    """
    Finds the start, peak, and end frame indices of the kick motion, not just
    the single peak frame. Needed for any metric that depends on *how the body
    moves over time* (foot pivot, retraction speed, hand motion, shoulder turn)
    rather than a single instant.

    Approach: compute kicking-ankle height relative to hip for every frame,
    establish a baseline from the first few frames (assumed to be a ready
    stance), then walk backward/forward from the peak frame while height stays
    above baseline + height_margin.

    Returns (start_idx, peak_idx, end_idx, low_confidence_window), or
    (None, None, None, None) if no usable kick was found. low_confidence_window
    is True when the threshold-crossing search couldn't find a real start
    before the peak (e.g. a very fast kick, or the video starting mid-motion)
    and had to fall back to a fixed minimum window -- callers should treat
    metrics computed over such a window more skeptically.
    """
    side = kicking_side.upper()
    heights = {}  # frame_idx -> height
    for f in all_frames:
        lm = f["landmarks"]
        if lm is None:
            continue
        try:
            ankle_y = lm[f"{side}_ANKLE"][1]
            hip_y = lm[f"{side}_HIP"][1]
            ankle_vis = lm[f"{side}_ANKLE"][2]
            if ankle_vis < 0.4:
                continue
            heights[f["frame_idx"]] = hip_y - ankle_y  # positive = ankle above hip
        except KeyError:
            continue

    if not heights:
        return None, None, None, None

    sorted_idxs = sorted(heights.keys())
    baseline_sample = [heights[i] for i in sorted_idxs[: min(10, len(sorted_idxs))]]
    baseline = sorted(baseline_sample)[len(baseline_sample) // 2]  # median

    peak_idx = max(heights, key=heights.get)
    threshold = baseline + height_margin

    if heights[peak_idx] < threshold:
        # No real kick detected -- ankle never rose meaningfully above stance baseline
        return None, None, None, None

    low_confidence_window = False

    start_idx = peak_idx
    found_start = False
    for i in sorted_idxs:
        if i > peak_idx:
            break
        if i not in heights:
            continue
        if heights[i] >= threshold:
            start_idx = i
            found_start = True
            break

    if not found_start or (peak_idx - start_idx) < min_window_frames:
        # Threshold crossing happened within one frame (fast kick / clip starts
        # mid-motion) -- fall back to a fixed minimum window so downstream
        # "change over time" metrics have something non-zero to measure, but
        # flag it so callers know this window is a rough approximation.
        candidate_start = max(min(sorted_idxs), peak_idx - min_window_frames)
        if candidate_start in heights:
            start_idx = candidate_start
        else:
            # nearest available frame index at/after candidate_start
            later = [i for i in sorted_idxs if i >= candidate_start and i <= peak_idx]
            start_idx = later[0] if later else peak_idx
        low_confidence_window = True

    end_idx = peak_idx
    found_end = False
    for i in sorted_idxs:
        if i < peak_idx:
            continue
        end_idx = i
        if heights[i] < threshold:
            found_end = True
            break

    if not found_end or (end_idx - peak_idx) < min_window_frames:
        candidate_end = min(max(sorted_idxs), peak_idx + min_window_frames)
        later = [i for i in sorted_idxs if i <= candidate_end and i >= peak_idx]
        if later:
            end_idx = later[-1]
        low_confidence_window = low_confidence_window or not found_end

    return start_idx, peak_idx, end_idx, low_confidence_window


def compute_foot_pivot(all_frames, start_idx, peak_idx, kicking_side="right"):
    """
    Measures how much the standing (base) foot pivots during the kick --
    the classic "turn your foot over, get on your toes" cue. Compares the
    standing foot's heel-to-toe line angle at the start of the kick window
    vs. at the peak.

    EXPERIMENTAL: validated against only two real clips so far (both showed
    ~47-52 degrees of pivot on confirmed-good technique). Not yet tested
    against a confirmed-bad (foot not pivoting) example. Treat as a rough
    signal, not a validated verdict -- see calibration notes in README.

    Returns a dict with pivot_deg and visibility, or None values if landmarks
    are missing.
    """
    side = kicking_side.upper()
    other = "LEFT" if side == "RIGHT" else "RIGHT"

    def get_frame(idx):
        return next((f for f in all_frames if f["frame_idx"] == idx), None)

    f_start = get_frame(start_idx)
    f_peak = get_frame(peak_idx)
    if f_start is None or f_peak is None or f_start["landmarks"] is None or f_peak["landmarks"] is None:
        return {"pivot_deg": None, "visibility": None}

    lm_start, lm_peak = f_start["landmarks"], f_peak["landmarks"]

    try:
        angle_start = line_angle(lm_start[f"{other}_HEEL"][:2], lm_start[f"{other}_FOOT_INDEX"][:2])
        angle_peak = line_angle(lm_peak[f"{other}_HEEL"][:2], lm_peak[f"{other}_FOOT_INDEX"][:2])
        pivot_deg = abs(angle_peak - angle_start)
        # Handle angle wraparound (e.g. -179 vs 179 is actually a 2-degree difference, not 358)
        if pivot_deg > 180:
            pivot_deg = 360 - pivot_deg
        visibility = min(
            lm_start[f"{other}_HEEL"][2], lm_start[f"{other}_FOOT_INDEX"][2],
            lm_peak[f"{other}_HEEL"][2], lm_peak[f"{other}_FOOT_INDEX"][2],
        )
        return {"pivot_deg": pivot_deg, "visibility": visibility}
    except KeyError:
        return {"pivot_deg": None, "visibility": None}


# Provisional -- based on exactly two confirmed-good data points (47.1 and
# 52.3 degrees). Wide margin deliberately, since this is not yet validated
# against a confirmed-bad example. Expect to tighten or correct this once
# more clips (especially ones with a known pivot problem) are run through it.
FOOT_PIVOT_PROVISIONAL_RANGE = (30, 90)


def find_peak_kick_frame(all_frames, kicking_side="right"):
    """
    Heuristic: the peak of the kick is the frame where the kicking-side ankle
    is at its highest point (lowest normalized y-value) relative to the hip.
    Only considers frames with usable landmark data.
    Returns the frame index, or None if no usable frames found.
    """
    side = kicking_side.upper()
    best_idx = None
    best_height = None  # smaller (more negative) is higher

    for f in all_frames:
        lm = f["landmarks"]
        if lm is None:
            continue
        try:
            ankle_y = lm[f"{side}_ANKLE"][1]
            hip_y = lm[f"{side}_HIP"][1]
            height = ankle_y - hip_y  # negative when ankle is above hip
            ankle_vis = lm[f"{side}_ANKLE"][2]
            if ankle_vis < 0.4:
                continue
            if best_height is None or height < best_height:
                best_height = height
                best_idx = f["frame_idx"]
        except KeyError:
            continue

    return best_idx


# Reference ranges -- STARTING POINT ONLY.
# These are rough approximations based on general Muay Thai coaching cues, not
# yet validated against real coach-reviewed footage. Treat Track 1 as the step
# that tells you whether these need adjusting.
REFERENCE_RANGES = {
    "hip_line_angle_deg": (25, 55),       # degrees of hip-line tilt from horizontal at peak kick
    "standing_knee_angle_deg": (145, 170), # slight bend, not locked
    "torso_lean_deg": (0, 45),             # degrees of backward lean from vertical
                                             # (widened from 0-20 after a confirmed-good
                                             # kick measured 40.4 -- see calibration log)
    "guard_drop_normalized": (-0.15, 0.05), # negative = guard hand near/above shoulder height
}


def generate_feedback(metrics):
    """
    Turns raw metrics into plain-language coaching cues by comparing against
    REFERENCE_RANGES. Returns a list of (metric, status, message) tuples.
    """
    feedback = []

    VISIBILITY_THRESHOLD = 0.5

    checks = [
        ("hip_line_angle_deg", "hip_line_visibility", "hip rotation",
         "Your hips are rotating well through the kick.",
         "Your hip rotation looks short — try turning your standing hip over more as you kick.",
         "Your hips may be over-rotating — check you're not spinning off balance."),
        ("standing_knee_angle_deg", "standing_knee_visibility", "standing leg",
         "Good bend in your standing leg for the pivot.",
         "Your standing leg looks locked straight — a slight bend helps you pivot through the kick.",
         "Your standing leg may be bending too much — check your balance isn't collapsing."),
        ("torso_lean_deg", "torso_visibility", "torso lean",
         "Torso stays upright and controlled.",
         "Good — minimal backward lean.",
         "You're leaning back noticeably — that leaks power out of the kick and drops your guard."),
        ("guard_drop_normalized", "guard_visibility", "guard position",
         "Guard hand stays up near your head.",
         "Guard hand stays up near your head.",
         "Your guard hand is dropping during the kick — keep it near your chin."),
    ]

    for key, vis_key, label, ok_msg, low_msg, high_msg in checks:
        value = metrics.get(key)
        visibility = metrics.get(vis_key)
        if value is None:
            feedback.append((label, "unmeasured", f"Couldn't reliably measure {label} in this clip."))
            continue
        if visibility is not None and visibility < VISIBILITY_THRESHOLD:
            feedback.append((label, "unmeasured",
                              f"Tracking wasn't reliable enough on {label} in this clip to give confident feedback "
                              f"(landmark visibility {visibility:.0%}) — try a clearer angle or better lighting."))
            continue
        lo, hi = REFERENCE_RANGES[key]
        if lo <= value <= hi:
            feedback.append((label, "ok", ok_msg))
        elif value < lo:
            feedback.append((label, "low", low_msg))
        else:
            feedback.append((label, "high", high_msg))

    return feedback


def generate_experimental_feedback(all_frames, kicking_side="right"):
    """
    Separate from generate_feedback() on purpose: this covers metrics that
    are NOT yet validated the way hip rotation / standing leg / torso lean
    are. Currently just foot pivot. Returned feedback should always be
    visually/textually distinguished from the core validated feedback in
    any UI that shows it -- don't blend "confirmed" and "experimental"
    verdicts together.

    Returns a list of (label, status, message) tuples, same shape as
    generate_feedback(), plus the raw window/pivot data for debugging.
    """
    start, peak, end, low_confidence_window = find_kick_window(all_frames, kicking_side=kicking_side)
    feedback = []

    if start is None:
        feedback.append(("foot pivot", "unmeasured",
                          "Couldn't isolate the kick motion in this clip to measure foot pivot."))
        return feedback, None

    pivot_result = compute_foot_pivot(all_frames, start, peak, kicking_side=kicking_side)
    pivot_deg = pivot_result["pivot_deg"]
    visibility = pivot_result["visibility"]

    if pivot_deg is None or visibility is None:
        feedback.append(("foot pivot", "unmeasured", "Couldn't reliably track the standing foot in this clip."))
        return feedback, pivot_result

    if visibility < 0.5:
        feedback.append(("foot pivot", "unmeasured",
                          f"Standing foot wasn't tracked reliably enough (visibility {visibility:.0%})."))
        return feedback, pivot_result

    confidence_note = " (window timing was approximated for this clip -- treat as rougher than usual)" \
        if low_confidence_window else ""

    lo, hi = FOOT_PIVOT_PROVISIONAL_RANGE
    if lo <= pivot_deg <= hi:
        feedback.append(("foot pivot", "ok",
                          f"Standing foot pivots well ({pivot_deg:.0f}°){confidence_note}."))
    else:
        feedback.append(("foot pivot", "low",
                          f"Standing foot isn't pivoting much ({pivot_deg:.0f}°, expected roughly {lo}-{hi}°) "
                          f"-- try turning your base foot over so the heel points toward the target{confidence_note}."))

    return feedback, pivot_result