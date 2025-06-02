# 🌱 Paragon ADHD-Grow on MNIST

This project demonstrates **adaptive neural growth using ADHD-based scoring** on the MNIST dataset, powered by the [Paragon](https://github.com/openfluke/paragon) AI framework.

We simulate **biologically-inspired neuroplasticity** by extracting a subnetwork (micro-network), injecting new layers, evaluating performance using a confidence-centric ADHD score, and surgically reattaching improved structures to the parent network.

## 🚀 Overview

- **Framework**: [Paragon](https://github.com/openfluke/paragon) (modular neural substrate in Go with WebGPU support)
- **Dataset**: MNIST handwritten digits
- **Objective**: Grow a better network by inserting new layers where needed
- **Scoring**: ADHD Score — a proxy for confidence-based correctness
- **Growth Mechanism**:
  - Extract micro-network around a checkpoint layer
  - Insert a new hidden layer with random width/activation
  - Train briefly, evaluate ADHD score
  - Keep the best candidate and surgically reattach it if it improves performance

## 🧠 Highlights

- Supports `Grow()` calls that autonomously mutate and verify network structure
- Can distribute micro-network training across devices
- GPU-accelerated via WebGPU (cross-platform: Vulkan, Metal, DirectX12)
- Modular architecture: Easily test custom growth logic or agent integration

## 🛠 Usage

Clone the repo and run the demo:

```bash
go run .
```

Example output:

```
✅ Network improved from 41.42 → 54.97 via Grow()
🚀 Network successfully improved by Grow()!
```

## 📦 Dependencies

- Go 1.21+
- Paragon (see `go.mod`)
- MNIST dataset (`./data/mnist/` with idx files)

## 🔬 Core Concepts

| Component            | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| `Grow()`             | Entry point for structural improvement search                       |
| `MicroNetwork`       | Lightweight subnet from input → checkpoint → output                 |
| `TryImprovement`     | Randomly adds a layer and evaluates ADHD gain                       |
| `ReattachToOriginal` | Inserts new layer back into full network if it improves performance |
| `EvaluateModel`      | Measures ADHD-based confidence score across predictions             |

## 🧪 Evaluation Metrics

- **ADHD Score**: Penalizes low-confidence correct predictions, rewards high-confidence correct ones
- **Composite Score**: Blended accuracy + confidence evaluation
- **Deviation Buckets**: Classifies errors by severity (0–10%, 10–20%, ..., 100%+)
- **Worst 5 Samples**: Diagnoses failure cases

## 🌍 Future Directions

- Distributed candidate training
- Evolutionary layer architectures
- Agent-controlled growth loops
- Integration with OpenFluke's Biofoundry and Pilot systems

---

Built with 🧠 by [@openfluke](https://github.com/openfluke) — let neural networks grow themselves.
