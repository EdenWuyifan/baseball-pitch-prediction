#!/usr/bin/env python3
"""
Naive Physics-Based Baseline for Pitch Classification
======================================================
This script uses a simple ballistic trajectory model to predict where a pitch
will cross home plate, then classifies it as strike/ball and predicts the zone.

NO MACHINE LEARNING - Pure physics-based heuristics.

Approach:
1. Project the ball's trajectory from release point to home plate
2. Use release position, speed, and break (pfx_x, pfx_z) to estimate plate crossing
3. Compare to strike zone boundaries to classify strike/ball
4. Map to MLB Gameday zone (1-14)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class NaivePhysicsPredictor:
    """
    Simple physics-based predictor that can be calibrated with training labels.
    """
    
    def __init__(self):
        self.PLATE_WIDTH = 17.0 / 12.0  # Home plate width in feet (17 inches)
        self.BALL_RADIUS = 1.5 / 12.0   # Baseball radius in feet (~1.5 inches)
        self.MOUND_TO_PLATE = 60.5      # Distance from mound to plate (feet)
        self.MPH_TO_FPS = 1.46667       # Miles per hour to feet per second
        
        # Calibration parameters (learned from training data)
        self.x_scale = 1.0
        self.x_offset = 0.0
        self.z_scale = 1.0
        self.z_offset = 0.0
    
    # ------------------------------------------------------------------ #
    # Physics model
    # ------------------------------------------------------------------ #
    def _predict_plate_crossing_raw(self, row):
        """Raw physics prediction without calibration."""
        release_x = row['release_pos_x']
        release_y = row['release_pos_y']
        release_z = row['release_pos_z']
        effective_speed = row['effective_speed']
        pfx_x = row['pfx_x'] / 12.0  # inches -> feet
        pfx_z = row['pfx_z'] / 12.0  # inches -> feet
        
        distance_to_plate = self.MOUND_TO_PLATE - release_y
        speed_fps = effective_speed * self.MPH_TO_FPS
        time_of_flight = max(distance_to_plate / max(speed_fps, 1e-3), 1e-3)
        
        # Horizontal: release position + total break
        predicted_x = release_x + pfx_x
        
        # Vertical: release height + gravity + break + aiming correction
        gravity_drop = -0.5 * 32.2 * (time_of_flight ** 2)
        trajectory_correction = distance_to_plate * 0.02  # pitchers aim upward
        predicted_z = release_z + gravity_drop + pfx_z + trajectory_correction
        
        return predicted_x, predicted_z
    
    def predict_plate_crossing(self, row):
        """Apply calibration to the raw physics prediction."""
        raw_x, raw_z = self._predict_plate_crossing_raw(row)
        calibrated_x = raw_x * self.x_scale + self.x_offset
        calibrated_z = raw_z * self.z_scale + self.z_offset
        return calibrated_x, calibrated_z
    
    # ------------------------------------------------------------------ #
    # Calibration utilities
    # ------------------------------------------------------------------ #
    def calibrate(self, df_train):
        """
        Fit linear corrections that align physics predictions with Statcast labels.
        """
        if 'plate_x' not in df_train or 'plate_z' not in df_train:
            print("  Skipping calibration: plate_x/plate_z not found.")
            return
        
        print("  Calibrating physics model on training labels...")
        raw_preds = df_train.apply(
            lambda row: pd.Series(self._predict_plate_crossing_raw(row), index=['raw_x', 'raw_z']),
            axis=1
        )
        
        self.x_scale, self.x_offset = self._fit_linear(raw_preds['raw_x'], df_train['plate_x'])
        self.z_scale, self.z_offset = self._fit_linear(raw_preds['raw_z'], df_train['plate_z'])
        
        calibrated_x = raw_preds['raw_x'] * self.x_scale + self.x_offset
        calibrated_z = raw_preds['raw_z'] * self.z_scale + self.z_offset
        
        x_mae = np.abs(calibrated_x - df_train['plate_x']).mean()
        z_mae = np.abs(calibrated_z - df_train['plate_z']).mean()
        
        print(f"    X mapping: {self.x_scale:.3f} * raw + {self.x_offset:.3f} (MAE {x_mae:.3f} ft)")
        print(f"    Z mapping: {self.z_scale:.3f} * raw + {self.z_offset:.3f} (MAE {z_mae:.3f} ft)")
    
    @staticmethod
    def _fit_linear(pred, target):
        """Return slope and intercept for target ≈ slope * pred + intercept."""
        if pred.std() < 1e-3:
            return 1.0, 0.0
        slope, intercept = np.polyfit(pred, target, 1)
        return slope, intercept
    
    def classify_pitch(self, plate_x, plate_z, sz_top, sz_bot):
        """
        Classify pitch as strike or ball based on strike zone.
        
        Args:
            plate_x: Horizontal crossing location
            plate_z: Vertical crossing location
            sz_top: Top of strike zone
            sz_bot: Bottom of strike zone
            
        Returns:
            str: 'strike' or 'ball'
        """
        # Strike zone with ball radius expansion
        # Horizontal: ±(plate_width/2 + ball_radius)
        # Vertical: [sz_bot - ball_radius, sz_top + ball_radius]
        
        x_limit = (self.PLATE_WIDTH / 2.0) + self.BALL_RADIUS
        z_lower = sz_bot - self.BALL_RADIUS
        z_upper = sz_top + self.BALL_RADIUS
        
        is_strike = (abs(plate_x) <= x_limit) and (z_lower <= plate_z <= z_upper)
        
        return 'strike' if is_strike else 'ball'
    
    def predict_zone(self, plate_x, plate_z, sz_top, sz_bot):
        """
        Predict MLB Gameday zone (1-14).
        
        Zone layout (catcher's view, negative plate_x = catcher's left):
        1  2  3   <- high
        4  5  6   <- middle
        7  8  9   <- low
        11/12/13  -> shadow zones above
        14        -> shadow zone below / wide low
        """
        plate_half = self.PLATE_WIDTH / 2.0
        x_limit = plate_half + self.BALL_RADIUS
        z_lower = sz_bot - self.BALL_RADIUS
        z_upper = sz_top + self.BALL_RADIUS
        sz_height = sz_top - sz_bot
        
        in_zone = (abs(plate_x) <= x_limit) and (z_lower <= plate_z <= z_upper)
        
        if in_zone:
            zone_height = sz_height / 3.0
            zone_width = self.PLATE_WIDTH / 3.0
            
            if plate_z >= sz_bot + 2 * zone_height:
                row = 0  # high
            elif plate_z >= sz_bot + zone_height:
                row = 1  # mid
            else:
                row = 2  # low
            
            if plate_x < -zone_width / 2.0:
                col = 0  # left
            elif plate_x > zone_width / 2.0:
                col = 2  # right
            else:
                col = 1  # middle
            
            zone = row * 3 + col + 1
        else:
            if plate_z < z_lower:
                zone = 14
            elif plate_z > z_upper:
                if plate_x < -x_limit:
                    zone = 11
                elif plate_x > x_limit:
                    zone = 13
                else:
                    zone = 12
            else:
                # Outside horizontally but within vertical band
                upper_third = sz_top - sz_height * (1 / 3)
                if plate_z > upper_third:
                    zone = 13 if plate_x > 0 else 11
                else:
                    zone = 14
        
        return int(zone)
    
    def predict_single(self, row):
        """
        Predict pitch_class and zone for a single pitch.
        
        Args:
            row: DataFrame row with pitch features
            
        Returns:
            tuple: (pitch_class, zone)
        """
        # Predict plate crossing location
        pred_x, pred_z = self.predict_plate_crossing(row)
        
        # Classify pitch
        pitch_class = self.classify_pitch(pred_x, pred_z, row['sz_top'], row['sz_bot'])
        
        # Predict zone
        zone = self.predict_zone(pred_x, pred_z, row['sz_top'], row['sz_bot'])
        
        return pitch_class, zone
    
    def predict(self, df):
        """
        Predict pitch_class and zone for entire dataframe.
        
        Args:
            df: DataFrame with pitch features
            
        Returns:
            DataFrame: Original df with added 'pitch_class' and 'zone' columns
        """
        predictions = df.apply(self.predict_single, axis=1)
        df['pitch_class'] = predictions.apply(lambda x: x[0])
        df['zone'] = predictions.apply(lambda x: x[1])
        
        return df


def evaluate_predictions(df_true, df_pred):
    """
    Evaluate predictions using the competition metric.
    
    Args:
        df_true: DataFrame with ground truth
        df_pred: DataFrame with predictions
        
    Returns:
        dict: Dictionary with accuracy metrics
    """
    # Merge on file_name
    merged = df_true.merge(df_pred, on='file_name', suffixes=('_true', '_pred'))
    
    # Pitch class accuracy
    class_correct = (merged['pitch_class_true'] == merged['pitch_class_pred']).sum()
    class_accuracy = class_correct / len(merged)
    
    # Zone accuracy
    zone_correct = (merged['zone_true'] == merged['zone_pred']).sum()
    zone_accuracy = zone_correct / len(merged)
    
    # Combined metric: 70% class + 30% zone
    combined_score = 0.7 * class_accuracy + 0.3 * zone_accuracy
    
    return {
        'class_accuracy': class_accuracy,
        'zone_accuracy': zone_accuracy,
        'combined_score': combined_score,
        'n_samples': len(merged)
    }


def main():
    """
    Main function to run naive baseline predictions.
    """
    # Setup paths
    data_dir = Path('./data')
    train_file = data_dir / 'train_ground_truth.csv'
    test_file = data_dir / 'test_features.csv'
    template_file = data_dir / 'test_submission_template.csv'
    output_file = Path('./naive_baseline_submission.csv')
    
    print("=" * 60)
    print("Naive Physics-Based Baseline Predictor")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    
    # Initialize predictor
    print("\n[2/5] Initializing predictor...")
    predictor = NaivePhysicsPredictor()
    
    # Calibrate on training data (physics parameter fit)
    print("\n[3/5] Calibrating physics model...")
    predictor.calibrate(df_train)
    
    # Evaluate on training set (sanity check)
    print("\n[4/5] Evaluating on training set (sanity check)...")
    df_train_pred = predictor.predict(df_train.copy())
    
    metrics = evaluate_predictions(
        df_train[['file_name', 'pitch_class', 'zone']],
        df_train_pred[['file_name', 'pitch_class', 'zone']]
    )
    
    print(f"  Class Accuracy: {metrics['class_accuracy']:.4f}")
    print(f"  Zone Accuracy: {metrics['zone_accuracy']:.4f}")
    print(f"  Combined Score: {metrics['combined_score']:.4f}")
    
    # Generate test predictions
    print("\n[5/5] Generating test predictions...")
    df_test_pred = predictor.predict(df_test.copy())
    
    # Save submission
    submission = df_test_pred[['file_name', 'pitch_class', 'zone']].copy()
    submission.to_csv(output_file, index=False)
    print(f"  Submission saved to: {output_file}")
    print(f"  Total predictions: {len(submission)}")
    
    # Show sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions:")
    print("=" * 60)
    print(submission.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✓ Baseline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

