# Gemini Context: Tokkatot AI - Chicken Disease Detection System

This document provides instructional context about the `tokkatot_ai` project, a machine learning system for detecting chicken diseases in the **Tokkatot Smart Chicken Farming ecosystem**.

## 1. Project Overview

**Purpose:** `tokkatot_ai` is a safety-first ensemble AI system designed to detect chicken diseases (Salmonella, Coccidiosis, New Castle Disease) by analyzing fecal images. It is evolving from a static inspection tool into a **24/7 real-time monitoring system** for manure conveyor belts.

**Strategy:**
1.  **High Recall (Edge):** 24/7 scanning via EfficientNetB0 to catch any potential disease marker.
2.  **Zero False Positive (Cloud):** Full ensemble verification (EfficientNetB0 + DenseNet121 + Safety Vote) in the cloud for confirmed diagnosis.
3.  **Outbreak Prevention:** Automated alerts to Tokkatot web app to ensure 100% farm safety.

**Technology Stack:**
*   **Language:** Python (>=3.12 for dev, 3.9 for edge conversion)
*   **ML Framework:** PyTorch (Dev/Cloud), TensorFlow (Edge/TFLite)
*   **Infrastructure:** Raspberry Pi + Hailo AI (Edge), vCPU Cloud Server (Cloud), MQTT/HTTP (Sync)

## 2. Current Project State (01 Mar 2026)

*   **Training:** Completed. Ensemble model (`ensemble_model.pth`) achieves 99% accuracy.
*   **Edge Models:** EfficientNetB0 + YOLOv8n converted to TFLite and ready in `outputs/tflite/`.
*   **Cloud Model:** Full ensemble already available in `outputs/ensemble_model.pth` for cloud verification (99% accuracy).
*   **Next:** Hailo HEF compilation on Raspberry Pi, then `edge_app.py` + cloud API development.

## 3. Key Development Tasks (Next Steps)

### Real-Time Monitoring (24/7 Deployment)
1.  **Edge Application:** Develop `edge_app.py` for continuous camera stream analysis.
2.  **Detection & Crop:** Implement YOLOv8n to identify feces on the moving belt.
3.  **Anomaly Detection:** Use the ensemble logic to identify disease markers in real-time.
4.  **Farm Integration:** Connect to the Tokkatot Cloud via MQTT for alerting and long-term analytics.

## 4. Building and Running (Local Dev)

### Setup
The project uses `uv` or `pip` for dependency management. Pre-trained models must be downloaded and placed in the `outputs/` directory as described in the `README.md`.

1.  **Install Dependencies:**
    ```bash
    # Using uv (recommended)
    uv pip install -e .

    # For development tools (linter, testing)
    uv pip install -e ".[dev]"
    ```
2.  **Verify Setup:**
    This script checks if the environment is correctly configured.
    ```bash
    python setup_check.py
    ```

### Key Commands

The main entry point is `main.py`, which dispatches commands.

*   **Training:**
    Train both models from scratch. Monitor progress using TensorBoard.
    ```bash
    # Start training
    python main.py train

    # Resume from the last checkpoint
    python main.py train --resume

    # Monitor with TensorBoard
    tensorboard --logdir outputs/logs
    ```

*   **Inference:**
    Run a prediction on a single image using the trained ensemble model.
    ```bash
    python main.py test path/to/your/image.jpg
    ```

*   **Evaluation:**
    Evaluate the final ensemble model's performance on the entire test dataset. Results are saved to `outputs/evaluation/`.
    ```bash
    python main.py eval
    ```

## 3. Development Conventions

*   **Dependency Management:** Project dependencies are defined in `pyproject.toml`.
*   **Code Style:** The project uses `black` for code formatting and `flake8` for linting.
*   **Testing:** `pytest` is used for running tests.
*   **Real-Time Monitoring:** See [REALTIME_MONITORING.md](REALTIME_MONITORING.md) for instructions on implementing 24/7 monitoring on Raspberry Pi.
*   **Documentation:**
    *   The `README.md` is the primary source of detailed project information.
    *   Model performance and details are documented in separate `MODEL_CARD_*.md` files.
*   **Model Storage:**
    *   Best-performing individual model checkpoints are saved in `outputs/checkpoints/`.
    *   The final, combined `ensemble_model.pth` is saved in `outputs/`.
