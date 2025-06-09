package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/openfluke/paragon/v3"
	"github.com/openfluke/pilot"
	"github.com/openfluke/pilot/experiments"
)

const (
	modelPath         = "mnist_model_float32.json"
	dataPath          = "./data/mnist"
	epochs            = 3
	learningRate      = 0.01
	trainLimit        = 10
	logFilePath       = "neural_network_growth.log"
	maxGrowthAttempts = 25 // Reduced from 50 to 25
	initialBatchSize  = 16 // Reduced from 32 to 16
)

var layers = []struct{ Width, Height int }{
	{28, 28}, {32, 32}, {10, 1},
}
var activations = []string{"linear", "relu", "softmax"}
var fullyConnected = []bool{true, true, true}

type GrowthAttempt struct {
	AttemptNumber       int
	Timestamp           string
	CheckpointLayer     int
	BeforeScore         float64 // Train score
	AfterScore          float64 // Train score
	BeforeTestScore     float64 // Test score
	AfterTestScore      float64 // Test score
	LayerFound          bool
	LayerIndex          int
	LayerWidth          int
	LayerHeight         int
	LayerActivation     string
	NetworkLayersBefore int
	NetworkLayersAfter  int
	BatchSize           int
	Improvement         float64 // Train improvement
	TestImprovement     float64 // Test improvement
	Accepted            bool
}

func main() {
	fmt.Println("🚀 Systematic Neural Network Growth Engine")

	// Initialize log file
	logFile, err := os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		fmt.Printf("❌ Failed to open log file: %v\n", err)
		return
	}
	defer logFile.Close()

	// Write log header
	writeLogHeader(logFile)

	// Load MNIST data
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

	// 🔧 LIMIT DATASET SIZE FOR TESTING
	trainLimit := 1000
	testLimit := 200

	if len(trainInputs) > trainLimit {
		trainInputs = trainInputs[:trainLimit]
		trainTargets = trainTargets[:trainLimit]
		fmt.Printf("🔧 Limited training set to %d samples (was %d)\n", trainLimit, len(trainInputs))
	}

	if len(testInputs) > testLimit {
		testInputs = testInputs[:testLimit]
		testTargets = testTargets[:testLimit]
		fmt.Printf("🔧 Limited test set to %d samples (was %d)\n", testLimit, len(testInputs))
	}

	fmt.Printf("📊 Final dataset sizes: Train=%d, Test=%d\n", len(trainInputs), len(testInputs))
	printMemoryUsage("after data loading")

	// Build initial network
	nn := paragon.NewNetwork[float32](
		[]struct{ Width, Height int }{{28, 28}, {32, 32}, {10, 1}},
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

	fmt.Println("🧠 Initial Network Structure:")
	printNetworkShape(nn)

	// Initial evaluation
	fmt.Println("\n🧪 Initial Training Set Evaluation:")
	initialTrainScore := evaluateFullNetwork(nn, trainInputs, trainTargets)
	fmt.Printf("Initial Training ADHD Score: %.4f\n", initialTrainScore)

	fmt.Println("\n🧪 Initial Test Set Evaluation:")
	initialTestScore := evaluateFullNetwork(nn, testInputs, testTargets)
	fmt.Printf("Initial Test ADHD Score: %.4f\n", initialTestScore)

	// Log initial state
	logInitialState(logFile, nn, initialTrainScore, initialTestScore)

	// Growth parameters
	batchSize := initialBatchSize
	totalCores := runtime.NumCPU()
	maxThreads := max(1, totalCores/2) // Use half the cores to reduce memory usage
	if maxThreads > 4 {
		maxThreads = 4 // Cap at 4 threads maximum
	}

	// Start systematic growth
	fmt.Println("\n🌱 Starting Systematic Neural Network Growth...")
	fmt.Printf("📋 Max attempts: %d, Batch size: %d, Threads: %d\n", maxGrowthAttempts, batchSize, maxThreads)
	fmt.Printf("🎯 Strategy: Only trying to append 32x32 layers (any activation)\n")

	successfulGrowths := 0

	for attempt := 1; attempt <= maxGrowthAttempts; attempt++ {
		fmt.Printf("\n🔄 === Growth Attempt %d/%d ===\n", attempt, maxGrowthAttempts)
		printMemoryUsage(fmt.Sprintf("attempt %d start", attempt))

		// Try each possible checkpoint layer
		growthFound := false
		for checkpointLayer := 1; checkpointLayer < len(nn.Layers)-1; checkpointLayer++ {
			fmt.Printf("\n🎯 Trying checkpoint layer %d (of %d layers)\n", checkpointLayer, len(nn.Layers))

			// Prepare batch for growth attempt
			batchInputs, batchTargets := prepareBatch(trainInputs, trainTargets, batchSize, attempt)
			expectedLabels := extractLabels(batchTargets)

			// Get baseline FULL NETWORK ADHD scores before growth (both train and test)
			fmt.Printf("📊 Evaluating full network ADHD scores before growth...\n")
			baselineTrainScore := evaluateFullNetwork(nn, trainInputs, trainTargets)
			baselineTestScore := evaluateFullNetwork(nn, testInputs, testTargets)
			fmt.Printf("📊 Baseline TRAIN ADHD score: %.4f\n", baselineTrainScore)
			fmt.Printf("📊 Baseline TEST ADHD score: %.4f\n", baselineTestScore)

			// Create a copy of the network for testing
			networkCopy, err := cloneNetwork(nn)
			if err != nil {
				fmt.Printf("❌ Failed to clone network: %v\n", err)
				continue
			}

			// Attempt to grow the network copy using the small batch
			fmt.Printf("🧬 Testing growth: trying to append 32x32 layer at checkpoint %d...\n", checkpointLayer)
			layerFound := networkCopy.Grow(
				checkpointLayer,
				batchInputs,
				expectedLabels,
				20,        // candidates
				3,         // epochs
				0.01,      // learning rate
				1e-6,      // tolerance
				1.0, -1.0, // clip upper/lower
				32, 32, // FIXED: width range 32→32
				32, 32, // FIXED: height range 32→32
				[]string{"relu", "tanh", "leaky_relu", "sigmoid", "elu"},
				maxThreads,
			)

			// Evaluate the modified copy on FULL DATASET with ADHD scoring (both train and test)
			var afterTrainScore, afterTestScore float64
			var newLayerInfo LayerInfo

			if layerFound {
				fmt.Printf("🎉 32x32 layer found by Grow()! Now evaluating full network ADHD scores...\n")
				afterTrainScore = evaluateFullNetwork(networkCopy, trainInputs, trainTargets)
				afterTestScore = evaluateFullNetwork(networkCopy, testInputs, testTargets)
				newLayerInfo = getNewLayerInfo(nn, networkCopy)
				fmt.Printf("📊 NEW TRAIN ADHD score: %.4f (was %.4f) [%+.4f]\n", afterTrainScore, baselineTrainScore, afterTrainScore-baselineTrainScore)
				fmt.Printf("📊 NEW TEST ADHD score: %.4f (was %.4f) [%+.4f]\n", afterTestScore, baselineTestScore, afterTestScore-baselineTestScore)
				fmt.Printf("📏 New layer: Index %d, Size %dx%d, Activation: %s\n",
					newLayerInfo.Index, newLayerInfo.Width, newLayerInfo.Height, newLayerInfo.Activation)
			} else {
				afterTrainScore = baselineTrainScore
				afterTestScore = baselineTestScore
				fmt.Printf("😞 No 32x32 layer improvement found at checkpoint layer %d\n", checkpointLayer)
			}

			// Calculate improvement based on FULL network ADHD scores (train and test)
			trainImprovement := afterTrainScore - baselineTrainScore
			testImprovement := afterTestScore - baselineTestScore

			// Simple criteria: Accept if EITHER train OR test improves by >1.0
			trainGood := trainImprovement > 1.0
			testGood := testImprovement > 1.0
			accepted := layerFound && (trainGood || testGood)

			fmt.Printf("🔍 Acceptance check: Train %.4f>1.0=%v OR Test %.4f>1.0=%v\n",
				trainImprovement, trainGood, testImprovement, testGood)

			// Log this attempt
			growthAttempt := GrowthAttempt{
				AttemptNumber:       attempt,
				Timestamp:           time.Now().Format("2006-01-02 15:04:05"),
				CheckpointLayer:     checkpointLayer,
				BeforeScore:         baselineTrainScore,
				AfterScore:          afterTrainScore,
				BeforeTestScore:     baselineTestScore,
				AfterTestScore:      afterTestScore,
				LayerFound:          layerFound,
				NetworkLayersBefore: len(nn.Layers),
				BatchSize:           len(batchInputs),
				Improvement:         trainImprovement,
				TestImprovement:     testImprovement,
				Accepted:            accepted,
			}

			if layerFound {
				growthAttempt.LayerIndex = newLayerInfo.Index
				growthAttempt.LayerWidth = newLayerInfo.Width
				growthAttempt.LayerHeight = newLayerInfo.Height
				growthAttempt.LayerActivation = newLayerInfo.Activation
				growthAttempt.NetworkLayersAfter = len(networkCopy.Layers)
			} else {
				growthAttempt.NetworkLayersAfter = len(nn.Layers)
			}

			// Write to log file
			writeGrowthAttempt(logFile, growthAttempt)

			// If accepted, replace the main network
			if accepted {
				fmt.Printf("✅ ACCEPTING GROWTH! (Either train or test improved >1.0)\n")
				fmt.Printf("   📈 TRAIN improvement: %+.4f\n", trainImprovement)
				fmt.Printf("   📈 TEST improvement: %+.4f\n", testImprovement)
				*nn = *networkCopy
				successfulGrowths++
				growthFound = true

				// Log the updated network state
				fmt.Println("\n🧠 Updated Network Structure:")
				printNetworkShape(nn)

				logAcceptedGrowth(logFile, attempt, successfulGrowths, afterTrainScore, afterTestScore, trainImprovement, testImprovement)
				break // Move to next attempt
			} else {
				if layerFound {
					fmt.Printf("❌ REJECTING GROWTH:\n")
					fmt.Printf("   📉 TRAIN improvement: %+.4f (need >1.0) %s\n", trainImprovement, boolToIcon(trainGood))
					fmt.Printf("   📉 TEST improvement: %+.4f (need >1.0) %s\n", testImprovement, boolToIcon(testGood))
					fmt.Printf("   Need at least ONE to improve by >1.0\n")
				} else {
					fmt.Printf("❌ REJECTING - No layer found by Grow()\n")
				}
			}
		}

		if !growthFound {
			fmt.Printf("⚠️  No 32x32 layer improvement found in attempt %d\n", attempt)

			// If we haven't found growth in several attempts, try smaller batches
			if attempt%5 == 0 && batchSize > 16 {
				batchSize = max(16, batchSize-8)
				fmt.Printf("🔧 Reducing batch size to %d for better precision\n", batchSize)
			}
		}

		// Early termination if we've found several good layers
		if successfulGrowths >= 6 { // Reduced from 8 to 6
			fmt.Printf("🎯 Reached %d successful growths, terminating early\n", successfulGrowths)
			break
		}

		// Force garbage collection every few attempts to free memory
		if attempt%3 == 0 {
			runtime.GC()
			debug.FreeOSMemory()
		}
	}

	// Final evaluation and logging
	fmt.Println("\n🏁 === FINAL RESULTS ===")
	fmt.Printf("🌱 Successful 32x32 layer growths: %d\n", successfulGrowths)

	fmt.Println("\n🧠 Final Network Structure:")
	printNetworkShape(nn)

	fmt.Println("\n📊 Final Training Set Evaluation:")
	finalTrainScore := evaluateFullNetwork(nn, trainInputs, trainTargets)
	nn.PrintFullDiagnostics()

	fmt.Println("\n📊 Final Test Set Evaluation:")
	finalTestScore := evaluateFullNetwork(nn, testInputs, testTargets)
	nn.PrintFullDiagnostics()

	// Log final results
	logFinalResults(logFile, successfulGrowths, finalTrainScore, finalTestScore, initialTrainScore, initialTestScore)

	// Print growth history
	printGrowthHistory(nn)

	fmt.Printf("\n📄 Growth log saved to: %s\n", logFilePath)
}

type LayerInfo struct {
	Index      int
	Width      int
	Height     int
	Activation string
}

func cloneNetwork[T paragon.Numeric](original *paragon.Network[T]) (*paragon.Network[T], error) {
	data, err := original.MarshalJSONModel()
	if err != nil {
		return nil, err
	}

	clone := &paragon.Network[T]{}
	err = clone.UnmarshalJSONModel(data)
	if err != nil {
		return nil, err
	}

	// Copy important settings
	clone.WebGPUNative = original.WebGPUNative
	clone.Debug = original.Debug

	return clone, nil
}

func getNewLayerInfo[T paragon.Numeric](original, modified *paragon.Network[T]) LayerInfo {
	// Find the new layer by comparing layer counts and structures
	if len(modified.Layers) > len(original.Layers) {
		// New layer was added
		for i := 1; i < len(modified.Layers)-1; i++ { // Skip input and output layers
			if i >= len(original.Layers) ||
				modified.Layers[i].Width != original.Layers[i].Width ||
				modified.Layers[i].Height != original.Layers[i].Height {
				return LayerInfo{
					Index:      i,
					Width:      modified.Layers[i].Width,
					Height:     modified.Layers[i].Height,
					Activation: modified.Layers[i].Neurons[0][0].Activation,
				}
			}
		}
	}

	// Fallback: return info about the last hidden layer
	lastHidden := len(modified.Layers) - 2
	return LayerInfo{
		Index:      lastHidden,
		Width:      modified.Layers[lastHidden].Width,
		Height:     modified.Layers[lastHidden].Height,
		Activation: modified.Layers[lastHidden].Neurons[0][0].Activation,
	}
}

func prepareBatch(inputs, targets [][][]float64, batchSize, attempt int) ([][][]float64, [][][]float64) {
	// Use different batch each attempt for variety
	start := ((attempt - 1) * batchSize) % len(inputs)
	end := start + batchSize
	if end > len(inputs) {
		end = len(inputs)
	}

	return inputs[start:end], targets[start:end]
}

func extractLabels(targets [][][]float64) []float64 {
	labels := make([]float64, len(targets))
	for i := range targets {
		labels[i] = float64(paragon.ArgMax(targets[i][0]))
	}
	return labels
}

func evaluateNetworkOnBatch[T paragon.Numeric](nn *paragon.Network[T], inputs [][][]float64, expectedLabels []float64) float64 {
	actualLabels := make([]float64, len(inputs))

	for i, input := range inputs {
		nn.Forward(input)
		output := nn.GetOutput()
		actualLabels[i] = float64(paragon.ArgMax(output))
	}

	// Calculate accuracy percentage
	correct := 0
	for i := range expectedLabels {
		if expectedLabels[i] == actualLabels[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(expectedLabels)) * 100.0
}

func evaluateFullNetwork[T paragon.Numeric](nn *paragon.Network[T], inputs, targets [][][]float64) float64 {
	expected := make([]float64, len(inputs))
	actual := make([]float64, len(inputs))

	for i := range inputs {
		nn.Forward(inputs[i])
		out := nn.ExtractOutput()

		expected[i] = float64(paragon.ArgMax(targets[i][0]))
		actual[i] = float64(paragon.ArgMax(out))
	}

	nn.EvaluateFull(expected, actual)
	return nn.Performance.Score
}

func printNetworkShape[T paragon.Numeric](nn *paragon.Network[T]) {
	for i, layer := range nn.Layers {
		fmt.Printf("  Layer %d: %dx%d (%s)\n", i, layer.Width, layer.Height, layer.Neurons[0][0].Activation)
	}
}

func printGrowthHistory[T paragon.Numeric](nn *paragon.Network[T]) {
	if len(nn.GrowthHistory) == 0 {
		fmt.Println("📭 No growth history recorded.")
		return
	}

	fmt.Println("\n🌱 Growth History Summary:")
	for i, log := range nn.GrowthHistory {
		fmt.Printf("  [%d] ➕ Layer %d: %dx%d (%s) | Score %.2f → %.2f (+%.2f)\n",
			i+1, log.LayerIndex, log.Width, log.Height, log.Activation,
			log.ScoreBefore, log.ScoreAfter, log.ScoreAfter-log.ScoreBefore)
	}
}

// Logging functions
func writeLogHeader(file *os.File) {
	header := fmt.Sprintf(`
========================================
NEURAL NETWORK SYSTEMATIC GROWTH LOG
========================================
Started: %s
Strategy: Only appending 32x32 layers (any activation)
Batch Size: %d (for Grow() function)
Max Attempts: %d

NOTE: Before/After scores are FULL NETWORK ADHD SCORES on entire datasets
Grow() uses small batches to find layers, but acceptance is based on full network improvement

Acceptance Criteria: 
- Train improvement >1.0 OR Test improvement >1.0 (either one is fine!)

Columns:
Attempt | Timestamp | Checkpoint | Train_Before | Train_After | Train_Improvement | Test_Before | Test_After | Test_Improvement | Found | Accepted | LayerIdx | LayerSize | Activation | NetworkLayers

`, time.Now().Format("2006-01-02 15:04:05"), initialBatchSize, maxGrowthAttempts)

	file.WriteString(header)
}

func logInitialState(file *os.File, nn *paragon.Network[float32], trainScore, testScore float64) {
	msg := fmt.Sprintf("INITIAL STATE: %d layers, Train Score: %.4f, Test Score: %.4f\n",
		len(nn.Layers), trainScore, testScore)
	file.WriteString(msg)
}

func writeGrowthAttempt(file *os.File, attempt GrowthAttempt) {
	var layerInfo string
	if attempt.LayerFound {
		layerInfo = fmt.Sprintf("%d | %dx%d | %s",
			attempt.LayerIndex, attempt.LayerWidth, attempt.LayerHeight, attempt.LayerActivation)
	} else {
		layerInfo = "- | - | -"
	}

	acceptedStr := "NO"
	if attempt.Accepted {
		acceptedStr = "YES"
	}

	foundStr := "NO"
	if attempt.LayerFound {
		foundStr = "YES"
	}

	msg := fmt.Sprintf("%d | %s | %d | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f | %s | %s | %s | %d→%d\n",
		attempt.AttemptNumber,
		attempt.Timestamp,
		attempt.CheckpointLayer,
		attempt.BeforeScore,
		attempt.AfterScore,
		attempt.Improvement,
		attempt.BeforeTestScore,
		attempt.AfterTestScore,
		attempt.TestImprovement,
		foundStr,
		acceptedStr,
		layerInfo,
		attempt.NetworkLayersBefore,
		attempt.NetworkLayersAfter)

	file.WriteString(msg)
}

func logAcceptedGrowth(file *os.File, attempt, totalGrowths int, trainScore, testScore, trainImprovement, testImprovement float64) {
	msg := fmt.Sprintf("ACCEPTED GROWTH #%d (Attempt %d): TRAIN %.4f (%+.4f) | TEST %.4f (%+.4f)\n",
		totalGrowths, attempt, trainScore, trainImprovement, testScore, testImprovement)
	file.WriteString(msg)
}

func logFinalResults(file *os.File, growths int, finalTrain, finalTest, initialTrain, initialTest float64) {
	msg := fmt.Sprintf(`
========================================
FINAL RESULTS
========================================
Total Successful Growths: %d
Initial Scores - Train: %.4f, Test: %.4f
Final Scores   - Train: %.4f, Test: %.4f
Improvement    - Train: %+.4f, Test: %+.4f
Completed: %s
========================================
`, growths, initialTrain, initialTest, finalTrain, finalTest,
		finalTrain-initialTrain, finalTest-initialTest, time.Now().Format("2006-01-02 15:04:05"))

	file.WriteString(msg)
}

func printMemoryUsage(stage string) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("🧠 Memory [%s]: %.2f MB allocated, %.2f MB total\n",
		stage, float64(m.Alloc)/1024/1024, float64(m.Sys)/1024/1024)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func boolToIcon(b bool) string {
	if b {
		return "✅"
	}
	return "❌"
}
