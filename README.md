# 🎯 Video-Only Early-Flight Pitch Classification Challenge

## Overview
**Predict the Strike Zone… Before the Ball Gets There**

Baseball pitchers throw a ball over 60 feet toward home plate, and human umpires must decide—within milliseconds—whether a pitch is a strike or a ball. But what if we try to make that decision before the pitch even reaches the plate?

In this competition, your goal is to build a model that predicts whether a pitch will end up inside the strike zone using only a short video clip of the ball's early flight.

### Input Data Types
You will be given two types of clips:
- **Full Pitch clips** (1.2s) — roughly the entire pitch
- **Trimmed clips** (0.26–0.75s) — short physics-based clips aligned to ~80% of the ball's time-of-flight

### Task
For each clip, you must predict whether the pitch will or will not cross the strike zone, and which part of the zone, even though the ball has not yet reached the plate.

## 🎯 What Are You Predicting?

Your goal is simple: predict the final pitch outcome using only early-flight video.

### Ground Truth Source
Labels come from real Statcast trajectory measurements, not umpire calls.

### Key Parameters
Each pitch includes:
- `plate_x` — horizontal plate crossing location
- `plate_z` — vertical plate crossing location
- `sz_top` — top of batter's strike zone
- `sz_bot` — bottom of batter's strike zone

### Strike Determination
A pitch is labeled **strike** if the center of the baseball passes through the strike-zone volume (expanded slightly by the ball's radius ≈1.5 inches). Otherwise, it is labeled **ball**.

### MLB Gameday Strike Zone
MLB's standard strike-zone grid (catcher's view):

```
    Boxes 1–9  → true strike-zone
    Boxes 11–14 → near-border "shadow" zones
```

**Challenge**: Your model sees only the start of the ball's flight but must infer where it will end up.


## 📊 Evaluation

Submissions are evaluated using a **weighted multi-target accuracy**, combining:

### 1. Pitch Class Prediction (70% weight)
Predict whether the pitch ends up in the strike zone or not. For each test video, `pitch_class` must be one of:
- `strike`
- `ball`

### 2. Zone Prediction (30% weight)
Predict the exact MLB Gameday zone (1-14) where the pitch will end up.

### Zone Accuracy Formula
```
AccuracyZone = (1/N) * Σ I(ẑ_i = z_i)
```

### Final Competition Score
The final score is a weighted combination:
- **Class Accuracy**: 70% weight
- **Zone Accuracy**: 30% weight

**Scoring Examples:**
- If `pitch_class` is correct but `zone` is wrong → earn 0.7 for that row
- If both are correct → earn 1.0 (full credit)

The score ranges from 0 to 1, where 1.0 = perfect predictions on both targets.

## 📝 Submission Format

Your submission must be a CSV with one row per test video containing:

```csv
file_name,pitch_class,zone
pitch1.mp4,strike,5
pitch2.mp4,ball,14
pitch3.mp4,strike,3
...
```



## 📋 Dataset Description

### Ground Truth Labeling (Statcast Parameters)
Every pitch uses MLB's Statcast tracking to determine ground-truth strike vs ball.

#### Plate Crossing Location
- `plate_x`, `plate_z`: Define where the ball crossed the front plane of home plate
  - Positive `plate_x` → catcher's right
  - Negative `plate_x` → catcher's left
  - `plate_z` → height of the pitch at the plate

#### Strike Zone Dimensions
- `sz_top`, `sz_bot`: Personalized strike-zone height for each batter

#### Strike Criteria
A pitch is a **strike** if:
```
|plate_x| ≤ (17/24 + r) ∧ sz_bot - r ≤ plate_z ≤ sz_top + r
```
Where `r` ≈ 1.5 inches (baseball radius).

### Time-of-Flight (ToF): Why Clips Are So Short

Each trimmed test clip is **0.26–0.75 seconds** long, representing roughly **80% of the ball's Time-of-Flight**.

#### ToF Estimation Formula
```
Distance = 60.5 ft - release_extension
Time = Distance / (release_speed * 1.46667)  # mph to ft/s conversion
```

**Key Challenge**: Only early-flight motion is visible — the ball never reaches the plate in the video. This ensures the prediction task is genuinely difficult.

### Dataset Overview

#### Training Set
- `train_trimmed/` – 0.26–0.75s physics-aligned clips (80% of estimated ToF)

#### Test Set
- `test/` – trimmed clips only (same format as training)

#### Video Naming
All videos are anonymized as: `pitch1.mp4`, `pitch2.mp4`, `pitch3.mp4`, ...
### Available Features

#### Pitch Physics Parameters
- `release_speed`: Speed of the pitch at release (mph)
- `effective_speed`: Adjusted speed representing how fast the pitch "plays" due to extension
- `release_spin_rate`: Spin rate (RPM)
- `release_pos_x, release_pos_y, release_pos_z`: 3D coordinates of the ball at release
- `release_extension`: How far in front of the mound the pitcher releases the ball (feet) - used to calculate Time-of-Flight
- `pfx_x, pfx_z`: Measured horizontal/vertical break of the pitch (inches relative to a spinless trajectory)

#### Batter/Pitcher Context
- `stand`: Batter stance
  - `"L"` = left-handed batter
  - `"R"` = right-handed batter
- `p_throws`: Pitcher throwing hand
  - `"L"` = left-handed pitcher
  - `"R"` = right-handed pitcher

## 📁 Provided CSV Files

This competition includes **three CSV files** that define the complete dataset structure.

### 1. `train_ground_truth.csv` - Training Labels & Metadata
Contains all training labels + full metadata needed to train your model.

**Each row corresponds to one anonymized pitch video.**

| Column | Description |
|--------|-------------|
| `file_name` | Anonymized filename (e.g., `pitch1.mp4`) |
| `plate_x`, `plate_z` | True crossing location of the pitch at the plate (feet) |
| `sz_top`, `sz_bot` | Personalized top & bottom of the batter's strike zone (feet) |
| `release_speed` | Pitch velocity at release (mph) |
| `effective_speed` | Adjusted perceived velocity due to extension |
| `release_spin_rate` | Spin rate (RPM) |
| `release_pos_x, y, z` | 3D coordinates of pitch release |
| `release_extension` | Release point extension toward home plate (feet) |
| `pfx_x`, `pfx_z` | Horizontal & vertical pitch break (inches) |
| `stand` | Batter stance (`L` or `R`) |
| `p_throws` | Pitcher throwing hand (`L` or `R`) |
| **`pitch_class`** | **Label you must predict** (`strike` or `ball`) |
| **`zone`** | **MLB Gameday zone number (1–14)** |

### 2. `test_features.csv` - Test Metadata (No Labels)
Contains all available metadata **EXCEPT** the columns that would leak the true label.

**Removed columns:** `plate_x`, `plate_z`, `pitch_class`, `zone`

| Column | Description |
|--------|-------------|
| `file_name` | Anonymized filename (test video) |
| `sz_top`, `sz_bot` | Batter strike-zone limits |
| `release_speed` | Pitch speed (mph) |
| `effective_speed` | Adjusted speed |
| `release_spin_rate` | Spin rate (RPM) |
| `release_pos_x, y, z` | 3D release coordinates |
| `release_extension` | Extension toward the plate (ft) |
| `pfx_x`, `pfx_z` | Horizontal/vertical movement |
| `stand` | Batter stance (`L` / `R`) |
| `p_throws` | Pitcher throwing hand (`L` / `R`) |

### 3. `test_template.csv` - Submission Template
This is the file you will submit predictions to Kaggle with. Contains the three required columns:

| Column | Description |
|--------|-------------|
| `file_name` | Anonymized pitch video (e.g., `pitch7001.mp4`) |
| **`pitch_class`** | **Predict:** `strike` or `ball` |
| **`zone`** | **Predict:** MLB Gameday zone (1–14)<br>• Zones 1–9 → inside strike zone<br>• Zones 11–14 → just outside<br>• Must predict exact number |

**Example Submission:**
```csv
file_name,pitch_class,zone
pitch7001.mp4,strike,5
pitch7002.mp4,ball,14
pitch7003.mp4,strike,3
```

## 🎯 Final Summary: What You Must Predict

1. **`pitch_class`**: `strike` or `ball`
2. **`zone`**: Integer 1–14 representing MLB Gameday strike zone

**Both predictions must be included in your `test_template.csv` submission.**