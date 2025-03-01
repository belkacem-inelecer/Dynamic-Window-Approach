#  Dynamic Window Approach (DWA) for Robot Navigation  

![DWA Simulation](https://github.com/belkacem-inelecer/Dynamic-Window-Approach/blob/main/media/Capture%20d%E2%80%99%C3%A9cran%202025-01-29%20135510.png)

## Overview  
This project implements the **Dynamic Window Approach (DWA)** for real-time motion planning of a **differential-drive robot**. The robot navigates toward a goal **while avoiding obstacles**, selecting the optimal trajectory at each time step.  

##  How It Works  
The **DWA algorithm** predicts multiple future trajectories and evaluates them using a **cost function**, which considers:  
✅ **Goal Distance** – Moving efficiently toward the target   
✅ **Obstacle Avoidance** – Steering clear of collisions  
✅ **Smooth Motion** – Ensuring stable and feasible control   

The robot then selects and executes the best trajectory, repeating the process dynamically.  

##  Features  
✔️ **Real-time Path Planning** using DWA  
✔️ **Obstacle Avoidance** with dynamic trajectory evaluation  
✔️ **Smooth Navigation** with optimized velocity control  
✔️ **Simulation & Visualization** using `matplotlib`  

## Installation & Setup  
### 1️ Clone the Repository  
```bash
git clone https://github.com/yourusername/dwa-robot-navigation.git
cd dwa-robot-navigation
```
### 2️ Install Dependencies  
```bash
pip install numpy matplotlib scipy scikit-optimize
```
### 3️ Run the Simulation  
```bash
python diff_drive_dwa.py
```

## ⚙ Project Structure  
```
 dwa-robot-navigation
 ├── diff_drive_dwa.py     # Main script for DWA navigation
 ├── README.md             # Documentation
 ├── requirements.txt      # Dependencies
 ├── media                 # Contains simulation images/videos
     ├── dwa_output.png       # Example simulation output
```

## Goal & Obstacles  
- **Goal Position:** `(1.5, 1.5)`  
- **Obstacles:** Randomly placed in the environment  

## Demo Video  
🎥 [Watch the Simulation in Action](Insert-Video-Link-Here)  

## Future Improvements  
 Add **LIDAR-based obstacle detection**  
 Implement **real-world robot control (ROS integration)**  
 Enhance **trajectory optimization for smoother paths**  

---

## Contributing  
Feel free to fork this repository, improve the algorithm, and submit pull requests! Let's build smarter robots together.  

## Contact & Feedback  
For discussions and improvements, reach out via [LinkedIn](https://www.linkedin.com/in/belkacem-bekkour-253185192/).  


#Robotics #PathPlanning #DWA #AI #AutonomousNavigation 🚀  
```

---
