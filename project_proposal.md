# Dynamic Traffic Signal Control System

## 1. Problem Description

Urban traffic congestion remains a major challenge in modern cities, causing delays, increased fuel consumption, and higher pollution levels.  
Conventional traffic light systems often rely on fixed time cycles that fail to adapt to real-time traffic conditions.  

The goal of this project is to design an **Dynamic Traffic Signal Control System** that dynamically optimizes the flow of vehicles through intersections.  
The controller should minimize average waiting time, reduce journey duration, and handle special cases such as the prioritization of emergency vehicles safely and efficiently.

---

## 2. Proposed Solution

We propose a computational approaches: **Q-Learning** 

### 2.1 Reinforcement Learning (Q-Learning)

Q-Learning is a model-free reinforcement learning algorithm suitable for discrete control tasks such as traffic signal switching.  
The environment provides the agent with the following observations:
- Number of vehicles in each lane  
- Current traffic light states  
- Average waiting time per lane  
- Detection of emergency vehicles  

The agent receives **rewards** when traffic flow improves (e.g., reduced waiting time or shorter queues) and **penalties** when performance worsens (e.g., long queues or near-collisions).  
By interacting with the simulated environment over time, the agent learns an **optimal policy** that minimizes overall traffic congestion.


---

## 3. Why This Approach Is Suitable

Traffic light control is a **sequential decision-making problem** that fits well within the reinforcement learning framework.  

---

## 4. Experimental Design and Evaluation

### 4.1 Simulation Setup

Experiments will be conducted using a **traffic flow simulator**, which can be found <a href="https://github.com/BilHim/trafficSimulator" target="_blank">here</a>


### 4.2 Performance Metrics

System performance will be evaluated using:
- **Average waiting time** per vehicle  
- **Average journey time** through the intersection  
- **Queue length** across lanes  
- **Throughput rate** (vehicles per time unit)  
- **Number of stops** per vehicle  

---

## 5. Expected Outcomes

The expected deliverables are:
1. A fully functional simulation of an intelligent traffic light control system.  
2. A trained Q-Learning agent capable of dynamic optimization.  
3. A Genetic Algorithm benchmark for heuristic comparison.  
