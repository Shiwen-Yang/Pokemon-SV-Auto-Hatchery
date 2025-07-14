# 🎮 AutoPoke_v1: Pokémon Violet Egg Hatching Automation

This project automates egg collection and hatching in *Pokémon Scarlet and Violet* using **Python**, **OpenCV**, and **Bluetooth controller emulation** via the [`nxbt`](https://github.com/Brikwerk/nxbt) library, which interfaces with **BlueZ** on Linux.

Designed to run **headless** and continuously, the system integrates real-time computer vision, controller macros, and a custom state controller to navigate in-game menus and actions. The system has operated **24/7 for 6+ months** with >99% uptime.

---

## Project Goals

- Automate **egg collection** by simulating picnic setup, sandwich crafting, and basket checking
- Automate **egg hatching** by transferring eggs from storage and moving in-game to trigger hatching
- Achieve high runtime stability and robustness over long periods
- Support **full headless operation** via terminal scripts

---

## Key Features

- **Graph-based state controller** for navigating in-game menus
- **Computer vision** (OCR + HSV filtering) for detecting game state and screen regions
- **Bluetooth controller emulation** using `nxbt` + BlueZ (Linux only)
- **Multithreaded frame capture** for real-time screen analysis
- Fault-tolerant design with recovery logic for common failure cases

---

## System Overview

### Egg Collection Pipeline
- Starts picnic
- Crafts sandwich to boost egg production
- Locates egg basket using screen-based pixel positioning
- Collects eggs by checking the basket and dismissing dialogues

### Egg Hatching Pipeline
- Loads eggs from PC boxes into party
- Uses directional movement to accumulate in-game distance
- Detects and counts hatch animations to track progress
- Returns hatchlings to storage and repeats process

---

## Technologies Used

- **Python 3**
- **OpenCV** (image processing and ROI detection)
- **Pytesseract / OCR** (text recognition)
- **NumPy**, **threading**
- **nxbt** (Switch controller emulation)
