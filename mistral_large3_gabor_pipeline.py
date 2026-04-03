import argparse
import base64
import io
import json
import os
from pathlib import Path

"""Mistral-Large-3 Gabor orientation discrimination pipeline.

This script mimics a human psychophysics experiment where an observer judges whether
noisy Gabor patches are tilted clockwise (CW) or counterclockwise (CCW) from vertical,
across eccentricity and external-noise conditions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


ECCENTRICITIES_DEG = [0, 5, 10]
NOISE_LEVELS = ["low", "medium", "high"]
ORIENTATION_OFFSETS_DEG = [-8, -6, -4, -2, 2, 4, 6, 8]
NOISE_SIGMA = {
    "low": 0.05,
    "medium": 0.12,
    "high": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Gabor orientation discrimination pipeline on Mistral-Large-3 via CCV LiteLLM."
    )
    parser.add_argument("--model", type=str, default="Mistral-Large-3")
    parser.add_argument("--trials-per-cell", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=72)
    parser.add_argument("--pixels-per-degree", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=1950)
    parser.add_argument("--output-dir", type=str, default="results/mistral_large3_gabor")
    parser.add_argument(
        "--save-stimuli",
        action="store_true",
        help="If set, saves stimulus PNGs to output_dir/stimuli/",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def build_client() -> OpenAI:
    # Loads OPENAI_API_KEY from environment or .env, then points requests to CCV LiteLLM.
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to your environment or a .env file."
        )

    base_url = os.getenv("LITELLM_BASE_URL", "https://litellm.ccv.brown.edu")
    return OpenAI(api_key=api_key, base_url=base_url)


def make_gabor_patch(
    size: int,
    orientation_deg_from_x: float,
    spatial_freq_cycles_per_px: float = 0.06,
    sigma_px: float = 16.0,
    phase_rad: float = 0.0,
) -> np.ndarray:
    # Build a sinusoidal grating modulated by a 2D Gaussian envelope.
    coords = np.arange(size) - (size - 1) / 2.0
    x, y = np.meshgrid(coords, coords)

    theta = np.deg2rad(orientation_deg_from_x)
    x_theta = x * np.cos(theta) + y * np.sin(theta)

    sinusoid = np.cos(2 * np.pi * spatial_freq_cycles_per_px * x_theta + phase_rad)
    envelope = np.exp(-(x**2 + y**2) / (2 * sigma_px**2))

    gabor = sinusoid * envelope
    return gabor


def render_trial_image(
    offset_deg: float,
    eccentricity_deg: float,
    noise_level: str,
    side: int,
    image_size: int,
    patch_size: int,
    pixels_per_degree: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Neutral gray background (0..1 scale), similar to classic vision-task stimuli.
    image = np.full((image_size, image_size), 0.5, dtype=float)

    # Convention: vertical = 90° in image coordinates; +offset means CW from vertical.
    orientation_deg_from_x = 90.0 - offset_deg

    patch = make_gabor_patch(
        size=patch_size,
        orientation_deg_from_x=orientation_deg_from_x,
        spatial_freq_cycles_per_px=0.06,
        sigma_px=patch_size * 0.22,
        phase_rad=float(rng.uniform(0, 2 * np.pi)),
    )

    # Contrast scaling keeps the patch visible while still allowing noise to dominate at high sigma.
    patch = 0.5 + 0.42 * patch

    eccentricity_px = int(round(eccentricity_deg * pixels_per_degree))
    center_y = image_size // 2
    center_x = image_size // 2 + side * eccentricity_px

    half = patch_size // 2
    y0, y1 = center_y - half, center_y - half + patch_size
    x0, x1 = center_x - half, center_x - half + patch_size

    if y0 < 0 or x0 < 0 or y1 > image_size or x1 > image_size:
        raise ValueError("Patch placement is out of bounds. Adjust image size or eccentricity.")

    image[y0:y1, x0:x1] = patch

    sigma = NOISE_SIGMA[noise_level]
    image += rng.normal(loc=0.0, scale=sigma, size=image.shape)

    # Fixation cross at center.
    c = image_size // 2
    image[c - 10 : c + 11, c - 1 : c + 2] = 0.1
    image[c - 1 : c + 2, c - 10 : c + 11] = 0.1

    return np.clip(image, 0.0, 1.0)


def image_to_data_url(image: np.ndarray) -> str:
    buf = io.BytesIO()
    plt.imsave(buf, image, cmap="gray", vmin=0.0, vmax=1.0, format="png")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_trial_table(trials_per_cell: int, rng: np.random.Generator) -> pd.DataFrame:
    # Balanced factorial design over eccentricity x noise, with random offsets/sides per cell.
    rows = []
    for ecc in ECCENTRICITIES_DEG:
        for noise in NOISE_LEVELS:
            offsets = rng.choice(ORIENTATION_OFFSETS_DEG, size=trials_per_cell, replace=True)
            sides = rng.choice([-1, 1], size=trials_per_cell, replace=True)
            for offset, side in zip(offsets, sides):
                rows.append(
                    {
                        "eccentricity_deg": int(ecc),
                        "noise_level": noise,
                        "offset_deg": int(offset),
                        "side": int(side),
                    }
                )
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)


def parse_model_choice(text: str) -> str | None:
    if text is None:
        return None

    cleaned = text.strip().upper().replace("-", " ")
    if "COUNTERCLOCKWISE" in cleaned or "CCW" in cleaned:
        return "CCW"
    if "CLOCKWISE" in cleaned or "CW" in cleaned:
        return "CW"
    return None


def query_model_choice(client: OpenAI, model: str, image_data_url: str) -> tuple[str | None, str]:
    # Keep prompt intentionally constrained to enforce 2AFC-style output.
    prompt = (
        "You are doing a 2AFC visual discrimination task. "
        "A single Gabor patch is shown around a fixation cross. "
        "Judge whether the patch is tilted clockwise or counterclockwise relative to vertical. "
        "Respond with exactly one token: CW or CCW."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        temperature=0,
        reasoning_effort=None,
    )

    raw_text = response.choices[0].message.content or ""
    parsed = parse_model_choice(raw_text)
    return parsed, raw_text


def estimate_threshold_75(group: pd.DataFrame) -> float | None:
    # Coarse psychometric threshold: smallest |offset| where accuracy >= 0.75.
    curve = (
        group.groupby("abs_offset_deg", as_index=False)["correct"]
        .mean()
        .sort_values("abs_offset_deg")
        .reset_index(drop=True)
    )

    hits = curve[curve["correct"] >= 0.75]
    if hits.empty:
        return None
    return float(hits.iloc[0]["abs_offset_deg"])


def save_summary_plots(df: pd.DataFrame, output_dir: Path) -> None:
    psychometric = (
        df.groupby(["eccentricity_deg", "noise_level", "abs_offset_deg"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, ecc in zip(axes, ECCENTRICITIES_DEG):
        sub = psychometric[psychometric["eccentricity_deg"] == ecc]
        for noise in NOISE_LEVELS:
            sub_n = sub[sub["noise_level"] == noise].sort_values("abs_offset_deg")
            ax.plot(sub_n["abs_offset_deg"], sub_n["accuracy"], marker="o", label=noise)
        ax.set_title(f"Eccentricity = {ecc}°")
        ax.set_xlabel("|Orientation offset| (deg)")
        ax.set_ylim(0, 1.02)
        ax.axhline(0.75, linestyle="--", linewidth=1, color="gray")
    axes[0].set_ylabel("Model accuracy")
    axes[-1].legend(title="Noise")
    fig.suptitle("Mistral-Large-3 Orientation Discrimination")
    fig.tight_layout()
    fig.savefig(output_dir / "psychometric_curves.png", dpi=150)
    plt.close(fig)


def run() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stimuli_dir = output_dir / "stimuli"
    if args.save_stimuli:
        stimuli_dir.mkdir(parents=True, exist_ok=True)

    client = build_client()

    trials = build_trial_table(args.trials_per_cell, rng)
    if args.max_trials is not None:
        trials = trials.head(args.max_trials).copy()

    # Collect one row per trial so downstream analyses can be reproduced from CSV alone.
    records = []
    for i, row in tqdm(trials.iterrows(), total=len(trials), desc="Running trials"):
        offset = float(row["offset_deg"])
        ecc = float(row["eccentricity_deg"])
        noise = str(row["noise_level"])
        side = int(row["side"])

        stimulus = render_trial_image(
            offset_deg=offset,
            eccentricity_deg=ecc,
            noise_level=noise,
            side=side,
            image_size=args.image_size,
            patch_size=args.patch_size,
            pixels_per_degree=args.pixels_per_degree,
            rng=rng,
        )

        if args.save_stimuli:
            stim_path = stimuli_dir / f"trial_{i:04d}.png"
            plt.imsave(stim_path, stimulus, cmap="gray", vmin=0.0, vmax=1.0)

        data_url = image_to_data_url(stimulus)

        # Ground-truth label comes from the signed orientation offset.
        true_choice = "CW" if offset > 0 else "CCW"

        parsed_choice = None
        raw_text = ""
        api_error = None
        try:
            parsed_choice, raw_text = query_model_choice(client, args.model, data_url)
        except Exception as exc:
            api_error = str(exc)

        # If parsing fails, keep trial as missing rather than forcing an arbitrary label.
        correct = None
        if parsed_choice in {"CW", "CCW"}:
            correct = int(parsed_choice == true_choice)

        records.append(
            {
                "trial": int(i),
                "eccentricity_deg": ecc,
                "noise_level": noise,
                "offset_deg": offset,
                "abs_offset_deg": abs(offset),
                "side": side,
                "true_choice": true_choice,
                "model_choice": parsed_choice,
                "model_raw_text": raw_text,
                "correct": correct,
                "api_error": api_error,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "trial_results.csv", index=False)

    # Restrict psychometric summaries to trials with parseable CW/CCW responses.
    valid_df = df[df["correct"].notna()].copy()
    if valid_df.empty:
        summary = {
            "model": args.model,
            "base_url": os.getenv("LITELLM_BASE_URL", "https://litellm.ccv.brown.edu"),
            "n_trials": int(len(df)),
            "n_valid_trials": 0,
            "message": "No valid CW/CCW responses parsed.",
        }
    else:
        condition_acc = (
            valid_df.groupby(["eccentricity_deg", "noise_level"], as_index=False)["correct"]
            .mean()
            .rename(columns={"correct": "accuracy"})
        )

        thresholds = (
            valid_df.groupby(["eccentricity_deg", "noise_level"])  # type: ignore[arg-type]
            .apply(estimate_threshold_75)
            .reset_index(name="threshold_75_deg")
        )

        summary = {
            "model": args.model,
            "base_url": os.getenv("LITELLM_BASE_URL", "https://litellm.ccv.brown.edu"),
            "n_trials": int(len(df)),
            "n_valid_trials": int(len(valid_df)),
            "overall_accuracy": float(valid_df["correct"].mean()),
            "accuracy_by_condition": condition_acc.to_dict(orient="records"),
            "thresholds_75_by_condition": thresholds.to_dict(orient="records"),
        }

        save_summary_plots(valid_df, output_dir)
        condition_acc.to_csv(output_dir / "accuracy_by_condition.csv", index=False)
        thresholds.to_csv(output_dir / "thresholds_75_by_condition.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Run complete.")
    print(f"Saved trial-level data to: {output_dir / 'trial_results.csv'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
