package main

import (
	"fmt"
	"os"

	"github.com/openfluke/paragon"
	"github.com/openfluke/pilot"
	"github.com/openfluke/pilot/experiments"
)

const (
	modelPath    = "mnist_model_float32.json"
	dataPath     = "./data/mnist"
	epochs       = 3
	learningRate = 0.01
	trainLimit   = 10 // 🔧 How many samples to train on
)

var layers = []struct{ Width, Height int }{
	{28, 28}, {32, 32}, {10, 1},
}
var activations = []string{"linear", "relu", "softmax"}
var fullyConnected = []bool{true, true, true}

func main() {
	fmt.Println("Water can")
	mnist := experiments.NewMNISTDatasetStage("./data/mnist")
	exp := pilot.NewExperiment("MNIST", mnist)

	if err := exp.RunAll(); err != nil {
		fmt.Println("❌ Experiment failed:", err)
		os.Exit(1)
	}

	allInputs, allTargets, err := loadMNISTData("./data/mnist")
	if err != nil {
		fmt.Println("❌ Failed to load MNIST:", err)
		return
	}

	// Split into 80% training and 20% testing
	trainInputs, trainTargets, testInputs, testTargets := paragon.SplitDataset(allInputs, allTargets, 0.8)

	// Build network
	nn := paragon.NewNetwork[float32](
		[]struct{ Width, Height int }{{28, 28}, {16, 16}, {10, 1}},
		[]string{"linear", "relu", "softmax"},
		[]bool{true, true, true},
	)

	// Enable WebGPU
	nn.WebGPUNative = true
	if err := nn.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("❌ Failed to initialize WebGPU: %v\n", err)
		return
	}
	defer nn.CleanupOptimizedGPU()

	nn.Debug = false

	// Limit train/test to 100 samples each
	trainInputs = trainInputs[:min(100, len(trainInputs))]
	trainTargets = trainTargets[:min(100, len(trainTargets))]
	testInputs = testInputs[:min(100, len(testInputs))]
	testTargets = testTargets[:min(100, len(testTargets))]

	// Evaluate on training set
	fmt.Println("🧪 Training Set Evaluation:")
	evaluateSet(nn, trainInputs, trainTargets)
	nn.PrintFullDiagnostics()

	// Evaluate on test set
	fmt.Println("\n🧪 Test Set Evaluation:")
	evaluateSet(nn, testInputs, testTargets)
	nn.PrintFullDiagnostics()

	// Evaluate on test set
	fmt.Println("\n🧪 Test Set Evaluation:")
	evaluateSet(nn, testInputs, testTargets)
	nn.PrintFullDiagnostics()

	// Run ADHD-based growth for 3 iterations
	growNetworkADHD(nn, trainInputs, trainTargets, 3)

	// Final diagnostic on training set
	fmt.Println("\n📊 Final ADHD Diagnostic - Training Set:")
	evaluateSet(nn, trainInputs, trainTargets)
	nn.PrintFullDiagnostics()

	// Final diagnostic on test set
	fmt.Println("\n📊 Final ADHD Diagnostic - Test Set:")
	evaluateSet(nn, testInputs, testTargets)
	nn.PrintFullDiagnostics()

}

func evaluateSet[T paragon.Numeric](nn *paragon.Network[T], inputs, targets [][][]float64) {
	expected := make([]float64, len(inputs))
	actual := make([]float64, len(inputs))

	for i := range inputs {
		nn.Forward(inputs[i])
		out := nn.ExtractOutput()

		expected[i] = float64(paragon.ArgMax(targets[i][0]))
		actual[i] = float64(paragon.ArgMax(out))
	}

	nn.EvaluateFull(expected, actual)
}
