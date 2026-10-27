# Smart Warehouse Vision

A Python-based warehouse product recognition system using computer vision and OCR.

## Overview

Smart Warehouse Vision helps identify warehouse products by reading printed product labels and matching OCR results against a local inventory database.

The system supports IP camera and mobile camera input, with separate scanning workflows for each source.

## Features

- IP camera streaming through RTSP
- Mobile camera scanning through a web interface
- Product label text recognition using PaddleOCR
- SKU, brand, and series matching against SQLite inventory data
- Multi-frame OCR voting to improve recognition reliability
- Object-presence and scene-change detection
- Duplicate recording prevention
- Manual review workflow for unknown products
- Flask-based inventory dashboard

## Tech Stack

- Python
- OpenCV
- PaddleOCR
- Flask
- SQLite
- NumPy
- JavaScript

## Architecture

Camera input → Object-presence detection → ROI extraction → Image preprocessing → OCR → SKU matching → Inventory lookup → Result recording or manual review.

## Security and Demo Data

This repository should use synthetic or authorized demonstration data. Camera credentials, production database files, captured images, and private environment settings must not be committed.

## Getting Started

1. Install a compatible Python version.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Configure the required environment variables.
5. Initialize the demo database and run the application.

Refer to the project documentation for detailed setup instructions.

## Project Status

Personal portfolio project. Features and setup requirements may evolve as the project is improved.
