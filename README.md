
# SafeNet 🛡️ | DoS & DDoS Detection and Mitigation System

**SafeNet** is a cybersecurity simulation and defense framework focused on **real-time detection and mitigation of DoS/DDoS attacks**. It features:
- Custom Python-based Intrusion Detection System (IDS)
- Machine Learning for anomaly detection
- Software-Defined Networking (SDN) for dynamic traffic control
- Open-source attack simulation tools and dashboards

---

## 📌 Objectives

- ✅ Simulate real-world DoS and DDoS attacks in isolated environments.
- ✅ Analyze packet-level and flow-based behaviors of SYN, UDP, ICMP, and HTTP floods.
- ✅ Implement rule-based and ML-based anomaly detection (Random Forest, Decision trees).
- ✅ Deploy SDN with Mininet + RYU for dynamic flow mitigation.
- ✅ Provide real-time visualizations and alerting systems.

---

## 🛠️ Tech Stack

- **Python** – Core language for IDS, ML, and scripts
- **Scikit-learn** – ML models for anomaly detection
- **Scapy**, **tcpdump**, **Wireshark** – Traffic analysis
- **Mininet**, **RYU**, **POX** – SDN setup
- **Hping3**, **LOIC**, **HOIC** – Attack simulation
- **Colorama**, **Threading**, **JSON**, **CSV**, **Signal**, **Deque**, **Subprocess** – Support modules
- **Streamlit/Flask** – Dashboard (optional)

---

## 🧪 Key Features

- 🧬 **Flow-Based Feature Logging**: 84+ features logged per network flow
- ⚠️ **Real-Time Alerting**: Packet sniffing and rule-based detection engine
- 🤖 **Anomaly Detection**: Machine Learning (Isolation Forest, One-Class SVM)
- 🔁 **SDN Mitigation**: Auto-update of flow rules on attacks
- 📊 **Dashboard (Planned)**: Visualize attacks, predictions, and traffic behavior

---

## 🏗️ Architecture

```
Attacker Tools ──> Virtual Switch (Mininet/OVS) ──> SDN Controller (RYU)
         │                                       │
         └────> IDS (Flow Logger + ML Model) <───┘
                       │
                   Alert & Block
```

---


---

## 🧪 Evaluation & Testing

| Test Type                 | Status | Description |
|--------------------------|--------|-------------|
| Packet Sniffing          | ✅ Pass | Validated with Wireshark |
| Flow Feature Logging     | ✅ Pass | Logged to CSV as expected |
| Rule-Based Detection     | ✅ Pass | Detected SYN floods |
| ML Anomaly Detection     | ✅ Pass | Predicted attacks with high accuracy |
| SDN Mitigation           | ✅ Pass | Auto blocking via RYU integration |

---

## 📦 Deliverables

- ✅ Python-based IDS + Flow Tracker
- ✅ ML integration for intelligent detection
- ✅ Attack simulation scripts (SYN/UDP/ICMP floods)
- ✅ SDN setup using Mininet + RYU
- ✅ GitHub repo with setup instructions and test data
- 📊 Dashboard and visualization (Coming Soon)

---

## 📂 GitHub Repositories

- 🔗 [Safe-Net DDoS Detection GitHub ](https://github.com/vermaShivang/Safe-Net-DDos-detection-and-Mitigation)

---

## 👨‍💻 Team EchoStorm

| Name           | Role                        |
|----------------|-----------------------------|
| Shivang Verma | Team Lead – Attack Simulation, SDN |
| Vedansh Shinde | Network Monitoring & Flow Logging |
| Vishu Chauhan  | ML Detection & Visualization |

---

## 🧠 Future Work

- Integrate batch ML prediction for lower latency
- Deploy a real-time dashboard using Streamlit
- Expand ML with deep learning models (LSTM, Autoencoders)
- Extend to L3-L7 application layer attacks
- Automate SDN rule generation based on alerts

---


---

> **"Simulate. Detect. Defend." – SafeNet, your AI-powered shield against DDoS**
