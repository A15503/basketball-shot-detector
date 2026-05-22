def generate_feedback(elbow_angle, knee_angle, wrist_angle, elbow_alignment, hip_shoulder_diff):
    feedback = []
    score = 0

    if 80 <= elbow_angle <= 110:
        feedback.append(("✅", "Elbow Angle", f"{elbow_angle}° — Good L-shape at release!"))
        score += 1
    elif elbow_angle < 80:
        feedback.append(("⚠️", "Elbow Angle", f"{elbow_angle}° — Too bent, aim for 80-110°"))
    else:
        feedback.append(("⚠️", "Elbow Angle", f"{elbow_angle}° — Too straight, keep elbow bent"))

    if 100 <= knee_angle <= 140:
        feedback.append(("✅", "Knee Bend", f"{knee_angle}° — Good power from legs!"))
        score += 1
    elif knee_angle > 140:
        feedback.append(("⚠️", "Knee Bend", f"{knee_angle}° — Bend knees more for power"))
    else:
        feedback.append(("⚠️", "Knee Bend", f"{knee_angle}° — Knees too bent at release"))

    if wrist_angle < 40:
        feedback.append(("✅", "Wrist Follow Through", f"{wrist_angle}° — Great snap!"))
        score += 1
    else:
        feedback.append(("⚠️", "Wrist Follow Through", f"{wrist_angle}° — Snap wrist down more (goose neck)"))

    if elbow_alignment < 0.05:
        feedback.append(("✅", "Elbow Alignment", f"{elbow_alignment:.3f} — Elbow nicely under the ball!"))
        score += 1
    else:
        feedback.append(("⚠️", "Elbow Alignment", f"{elbow_alignment:.3f} — Elbow flaring out, tuck it in"))

    if hip_shoulder_diff < 0.1:
        feedback.append(("✅", "Body Alignment", f"{hip_shoulder_diff:.3f} — Good straight form!"))
        score += 1
    else:
        feedback.append(("⚠️", "Body Alignment", f"{hip_shoulder_diff:.3f} — Body twisted, square up"))

    return feedback, score