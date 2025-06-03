package main

import (
	"fmt"
	"os"
	"runtime"
	"time"

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
	/*trainInputs = trainInputs[:min(100, len(trainInputs))]
	trainTargets = trainTargets[:min(100, len(trainTargets))]
	testInputs = testInputs[:min(100, len(testInputs))]
	testTargets = testTargets[:min(100, len(testTargets))]*/

	// Evaluate on training set
	fmt.Println("🧪 Training Set Evaluation:")
	evaluateSet(nn, trainInputs, trainTargets)
	nn.PrintFullDiagnostics()

	// Evaluate on test set
	fmt.Println("\n🧪 Test Set Evaluation:")
	evaluateSet(nn, testInputs, testTargets)
	nn.PrintFullDiagnostics()

	// 🔁 Run Grow()
	fmt.Println("\n🌱 Running ADHD-Based Grow()...")
	expectedLabels := make([]float64, len(trainTargets))
	for i := range trainTargets {
		expectedLabels[i] = float64(paragon.ArgMax(trainTargets[i][0]))
	}

	waiting()

	// 🔁 Run Grow() in batches of 64
	fmt.Println("\n🌱 Running ADHD-Based Grow() in batches...")

	batchSize := 64
	numBatches := 3
	improved := false

	totalCores := runtime.NumCPU()
	maxThreads := int(0.8 * float64(totalCores))
	if maxThreads < 1 {
		maxThreads = 1
	}

	checkpointLayer := 1
	increaseSizeCount := 0
	increaseSizeTrigger := 5
	layerWidthSize := 16
	layerHeightSize := 16
	successfulGrowths := 0
	maxLayersToAdd := 5 // Limit how many layers to add total

	for batch := 0; batch < numBatches; batch++ {
		start := batch * batchSize
		end := (batch + 1) * batchSize
		if start >= len(trainInputs) {
			break
		}
		if end > len(trainInputs) {
			end = len(trainInputs)
		}

		batchInputs := trainInputs[start:end]
		batchTargets := trainTargets[start:end]

		expectedLabels := make([]float64, len(batchTargets))
		for i := range batchTargets {
			expectedLabels[i] = float64(paragon.ArgMax(batchTargets[i][0]))
		}

		// Make sure we don't try to checkpoint beyond valid layers
		maxCheckpointLayer := len(nn.Layers) - 2 // Can't checkpoint on output layer
		if checkpointLayer > maxCheckpointLayer {
			checkpointLayer = maxCheckpointLayer
		}

		fmt.Printf("\n🔁 Batch %d: Running Grow() on %d samples (checkpoint layer %d)\n",
			batch+1, len(batchInputs), checkpointLayer)

		if nn.Grow(
			checkpointLayer, // checkpointLayer
			batchInputs,     // testInputs
			expectedLabels,  // expectedOutputs
			150,             // numCandidates
			5,               // epochs
			0.01,            // learningRate
			1e-6,            // tolerance
			1.0, -1.0,       // clipUpper / clipLower
			layerWidthSize, layerWidthSize, // minWidth = 16, maxWidth = 16
			layerHeightSize, layerHeightSize, // minHeight = 16, maxHeight = 16
			[]string{"relu", "tanh", "leaky_relu", "sigmoid"}, // activationPool
			maxThreads,
		) {
			improved = true
			successfulGrowths++
			fmt.Printf("✅ Batch resulted in improvement! (Growth #%d)\n", successfulGrowths)

			// Only move to next checkpoint layer after every 2-3 successful growths
			// This gives each layer multiple chances to grow
			if successfulGrowths%2 == 0 && checkpointLayer < len(nn.Layers)-2 {
				checkpointLayer += 1
				fmt.Printf("🔄 Moving to checkpoint layer %d\n", checkpointLayer)
			}

			// Stop if we've added too many layers
			if successfulGrowths >= maxLayersToAdd {
				fmt.Printf("🛑 Reached maximum layer limit (%d), stopping growth\n", maxLayersToAdd)
				break
			}

		} else {
			fmt.Println("⚠️  No improvement in this batch.")
			increaseSizeCount += 1
			if increaseSizeCount > increaseSizeTrigger {
				increaseSizeCount = 0
				layerWidthSize += 1
				layerHeightSize += 1
				fmt.Printf("📏 Increased layer size to %dx%d\n", layerWidthSize, layerHeightSize)
			}

			// Try a different checkpoint layer if current one isn't working
			if increaseSizeCount%3 == 0 && checkpointLayer > 1 {
				checkpointLayer = max(1, checkpointLayer-1)
				fmt.Printf("🔄 Trying earlier checkpoint layer %d\n", checkpointLayer)
			}
		}

		// Show current network status
		fmt.Printf("🏗️  Current network: %d layers, latest growth at layer %d\n",
			len(nn.Layers), checkpointLayer+1)
	}

	fmt.Println("\n🧠 Final Network Structure:")
	printNetworkShape(nn)

	if improved {
		fmt.Println("🚀 Network successfully improved by Grow()!")
	} else {
		fmt.Println("⚡️  No improvement found during Grow().")
	}

	// Final diagnostic on training set
	fmt.Println("\n📊 Final ADHD Diagnostic - Training Set:")
	evaluateSet(nn, trainInputs, trainTargets)
	nn.PrintFullDiagnostics()

	// Final diagnostic on test set
	fmt.Println("\n📊 Final ADHD Diagnostic - Test Set:")
	evaluateSet(nn, testInputs, testTargets)
	nn.PrintFullDiagnostics()

	// Final diagnostic on test set
	fmt.Println("\n📊 Final ADHD Diagnostic - Test Set:")
	evaluateSet(nn, testInputs, testTargets)
	nn.PrintFullDiagnostics()

	// 🌱 Growth history
	printGrowthHistory(nn)

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

func printNetworkShape[T paragon.Numeric](nn *paragon.Network[T]) {
	for i, layer := range nn.Layers {
		fmt.Printf("Layer %d: %dx%d (%s)\n", i, layer.Width, layer.Height, layer.Neurons[0][0].Activation)
	}
}

func printGrowthHistory[T paragon.Numeric](nn *paragon.Network[T]) {
	if len(nn.GrowthHistory) == 0 {
		fmt.Println("📭 No growth history recorded.")
		return
	}

	fmt.Println("\n🌱 Growth History:")
	for i, log := range nn.GrowthHistory {
		fmt.Printf("  [%d] ➕ Added Layer %d → %dx%d (%s)\n", i+1, log.LayerIndex, log.Width, log.Height, log.Activation)
		fmt.Printf("      Score Before: %.2f → After: %.2f\n", log.ScoreBefore, log.ScoreAfter)
		fmt.Printf("      ⏰ Time: %s\n", log.Timestamp)
	}
}

func waiting() {
	fmt.Println("Waiting for 5 seconds...")
	time.Sleep(5 * time.Second)
	fmt.Println("Done waiting!")
}
