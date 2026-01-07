# Satellite-Imagery-Based-Property-Valuation

# 🏡 Multimodal Property Valuation using Tabular Data & Satellite Imagery Overview

This project builds a Multimodal Regression Pipeline to predict real estate property prices by combining structured tabular data with satellite imagery.
The goal is to evaluate whether environmental and neighborhood-level visual cues (green cover, road density, water proximity, urban layout) can enhance traditional valuation models.

The project goes beyond standard regression by integrating computer vision, feature engineering, and ensemble learning into a unified system.

## Dataset
#### 1. Tabular Data

Includes property-level attributes such as:

Structural features (bedrooms, bathrooms, living area, floors)

Quality and condition indicators

Location-based attributes (zipcode, views)

**Engineered features:**

basement_ratio, above_ratio

rlt_size, rlt_lot_size

quality_liv, quality_cond

luxury_bool, water_view, view_flag

house_life, ren_age, lot_util, bath_per_bed

Outliers are clipped, skewed variables transformed, and features normalized.

#### 2. Satellite Imagery

Collected programmatically using the Mapbox Static Images API

RGB images only

Geo-aligned using latitude–longitude

Consistent zoom and resolution

Stored and aligned using a unique property id

## Modeling Approach
#### 1. Image Models

Three CNN backbones were fine-tuned for image-only price regression:

**ResNet**

**EfficientNet-B3**

**ConvNeXt** (best image-only performer)

Image embeddings were extracted and reduced using PCA (256 dimensions).

#### 2. Tabular Models

**LightGBM** (LGBM) used as the primary tabular model

Strong baseline performance due to rich feature engineering

#### 3. Fusion Strategies

**Image-only models**: CNN → embeddings → regression

**Mid-level fusion**: Tabular features + PCA-reduced image embeddings → XGBoost (K-Fold CV)

**Late-level fusion**: Prediction-level ensembling (weighted averaging, stacking)

## Results Summary
**Model Strategy	Validation R²
Tabular-only (LightGBM)	0.8837
Image-only (ConvNeXt)	~0.43
Image Ensemble (PCA + LGBM)	~0.45
Mid-level Fusion (XGBoost)	~0.829
Late-level Fusion	Worse than baseline**

Key Insight:
Tabular data dominates property valuation. Satellite imagery provides complementary but limited signal and must be carefully integrated to avoid degrading performance.

## Explainability

**Grad-CAM** used on **ConvNeXt** to visualize image regions influencing predictions

Visual explanations highlight environmental cues such as greenery, roads, and open spaces

Tabular models remain interpretable via feature importance and domain-driven feature
