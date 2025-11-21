
1. **Assemble detections per pitch**
   - Ingest YOLO outputs like `ball_detections_test.csv`; keep `file_name`, `frame_index`, `class_name`, `x_min`…`y_max`.
   - Filter for `baseball`, `homeplate`, `rubber`; drop low-confidence duplicates via per-frame NMS.
   - Join with metadata rows (`file_name` is the key) so every clip carries both tabular features and detection sequences.

2. **Calibrate camera using static landmarks**
   - Use the `rubber` and `homeplate` boxes to estimate a homography from image pixels to a canonical catcher-facing plane (rubber origin, home plate target).
   - Convert their pixel dimensions into feet using the known mound-to-plate distance (60.5 ft minus `release_extension`) to recover per-video scale/orientation.
   - Persist per-video calibration parameters (homography, scale, confidence) for downstream feature computation.

3. **Derive ball-flight trajectories**
   - Collapse each `baseball` box to its center `(u, v)` and map through the homography to field coordinates `(x, z)` in feet.
   - Convert `frame_index` to elapsed time using the clip FPS (Ultralytics metadata or a manual probe) so every sample becomes `(t, x_t, z_t)`.
   - Smooth the sequence (Kalman or spline) to remove jitter and impute short gaps with constant-acceleration constraints.
   - Differentiate the smoothed curve to recover velocity/acceleration vectors and curvature measurements.

4. **Predict `plate_x` and `plate_z`**
   - Set the plate plane distance as `d_plate = 60.5 ft - release_extension` in the calibrated coordinate frame; the mound/rubber origin is `t = 0`.
   - Fit a low-order kinematic model (`x(t) = x0 + vx t + 0.5 ax t²`, `z(t) = z0 + vz t + 0.5 az t²`) or a state-space filter to the `(t, x_t, z_t)` samples using weighted least squares (weights = detection confidence).
   - Solve for `t_plate` where the fitted `z(t)` equals `d_plate`, clamp to observed time range, then evaluate `x(t_plate)` for the predicted lateral crossing.
   - Re-evaluate `z(t_plate)` (or use the fitted vertical component if 3D calibration is available) to obtain `plate_z`; propagate solver residuals to produce uncertainty features.
   - Fall back to metadata-only priors when detections are too sparse (e.g., <4 frames) and flag those cases via a mask feature.

5. **Engineer fused features**
   - From the trajectory: release angle, instantaneous speeds, horizontal/vertical break deltas, predicted `plate_x`, `plate_z`, distance from strike-zone center, intercept uncertainty.
   - From metadata: `sz_top`, `sz_bot`, `release_speed`, `effective_speed`, `release_spin_rate`, `release_pos_*`, `release_extension`, `pfx_x`, `pfx_z`, `stand`, `p_throws`.
   - Concatenate both feature sets (plus QA metrics like calibration confidence, detection counts) to form the training matrix; standardize/encode as needed.

6. **Modeling & inference**
   - Start with an interpretable model (LightGBM/XGBoost) on the fused tabular features for `pitch_class` and `zone`.
   - Optionally add a sequential head (GRU/Transformer) that consumes the normalized `(x, z)` track and merges with the static features at the penultimate layer.
   - During inference, run YOLO ➜ calibration ➜ trajectory ➜ feature fusion to produce predictions for every clip.