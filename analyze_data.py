#!/usr/bin/env python3
"""Quick analysis to understand the data distribution."""

import pandas as pd
import numpy as np

# Load training data
df = pd.read_csv('./data/train_ground_truth.csv')

print("=" * 70)
print("DATA ANALYSIS")
print("=" * 70)

print("\n1. CLASS DISTRIBUTION:")
print(df['pitch_class'].value_counts(normalize=True))

print("\n2. ZONE DISTRIBUTION:")
print(df['zone'].value_counts().sort_index())

print("\n3. PLATE CROSSING STATISTICS:")
print(f"plate_x: mean={df['plate_x'].mean():.3f}, std={df['plate_x'].std():.3f}")
print(f"         min={df['plate_x'].min():.3f}, max={df['plate_x'].max():.3f}")
print(f"plate_z: mean={df['plate_z'].mean():.3f}, std={df['plate_z'].std():.3f}")
print(f"         min={df['plate_z'].min():.3f}, max={df['plate_z'].max():.3f}")

print("\n4. STRIKE ZONE STATISTICS:")
print(f"sz_top:  mean={df['sz_top'].mean():.3f}, std={df['sz_top'].std():.3f}")
print(f"sz_bot:  mean={df['sz_bot'].mean():.3f}, std={df['sz_bot'].std():.3f}")

print("\n5. RELEASE POSITION STATISTICS:")
print(f"release_pos_x: mean={df['release_pos_x'].mean():.3f}, std={df['release_pos_x'].std():.3f}")
print(f"release_pos_y: mean={df['release_pos_y'].mean():.3f}, std={df['release_pos_y'].std():.3f}")
print(f"release_pos_z: mean={df['release_pos_z'].mean():.3f}, std={df['release_pos_z'].std():.3f}")

print("\n6. PHYSICS PARAMETERS:")
print(f"release_speed: mean={df['release_speed'].mean():.1f} mph")
print(f"pfx_x: mean={df['pfx_x'].mean():.2f} in, std={df['pfx_x'].std():.2f} in")
print(f"pfx_z: mean={df['pfx_z'].mean():.2f} in, std={df['pfx_z'].std():.2f} in")

# Check correlation between actual plate_x and release_pos_x
print("\n7. CORRELATION CHECKS:")
print(f"plate_x vs release_pos_x: {df[['plate_x', 'release_pos_x']].corr().iloc[0,1]:.3f}")
print(f"plate_z vs release_pos_z: {df[['plate_z', 'release_pos_z']].corr().iloc[0,1]:.3f}")
print(f"plate_x vs pfx_x: {df[['plate_x', 'pfx_x']].corr().iloc[0,1]:.3f}")
print(f"plate_z vs pfx_z: {df[['plate_z', 'pfx_z']].corr().iloc[0,1]:.3f}")

# Show some example rows
print("\n8. SAMPLE DATA (strikes):")
strikes = df[df['pitch_class'] == 'strike'].head(3)
print(strikes[['plate_x', 'plate_z', 'sz_top', 'sz_bot', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'zone']])

print("\n9. SAMPLE DATA (balls):")
balls = df[df['pitch_class'] == 'ball'].head(3)
print(balls[['plate_x', 'plate_z', 'sz_top', 'sz_bot', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'zone']])

