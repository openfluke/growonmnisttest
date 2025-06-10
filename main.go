package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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
	learningRate      = 0.5
	logFilePath       = "neural_network_growth.log"
	checkpointFile    = "growth_checkpoint.json"
	maxGrowthAttempts = 25
	initialBatchSize  = 16
	trainChunkSize    = 1000
	testChunkSize     = 200
	modelsDir         = "./models"
	globalMinWidth    = 128
	globalMaxWidth    = 128

	globalMinHeight = 1
	globalMaxHeight = 1

	clippingUp  = 1
	clippingLow = -1
)

var layers = []struct{ Width, Height int }{
	{28, 28}, {globalMinWidth, globalMinHeight}, {10, 1},
}
var activations = []string{"linear", "relu", "softmax"}
var fullyConnected = []bool{true, true, true}

type GrowthAttempt struct {
	AttemptNumber       int
	Timestamp           string
	CheckpointLayer     int
	BeforeScore         float64
	AfterScore          float64
	BeforeTestScore     float64
	AfterTestScore      float64
	LayerFound          bool
	LayerIndex          int
	LayerWidth          int
	LayerHeight         int
	LayerActivation     string
	NetworkLayersBefore int
	NetworkLayersAfter  int
	BatchSize           int
	Improvement         float64
	TestImprovement     float64
	Accepted            bool
}

type GrowthSession struct {
	ChunkIndex      int
	TrainStartIdx   int
	TrainEndIdx     int
	TestStartIdx    int
	TestEndIdx      int
	InitialScore    float64
	FinalScore      float64
	GrowthsAccepted int
	ModelsSaved     int
	Duration        time.Duration
}

type ProcessingCheckpoint struct {
	LastCompletedChunk int       `json:"last_completed_chunk"`
	TotalModelsSaved   int       `json:"total_models_saved"`
	LastModelPath      string    `json:"last_model_path"`
	StartTime          time.Time `json:"start_time"`
	LastUpdateTime     time.Time `json:"last_update_time"`
}

func main() {
	programStartTime := time.Now()
	fmt.Println("🚀 Systematic Neural Network Growth Engine - Full Dataset Processing")

	// Create models directory if it doesn't exist
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		fmt.Printf("❌ Failed to create models directory: %v\n", err)
		return
	}

	// Initialize log file
	logFile, err := os.OpenFile(logFilePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		fmt.Printf("❌ Failed to open log file: %v\n", err)
		return
	}
	defer logFile.Close()

	// Check for existing models and resume capability
	latestModel, startChunk := findLatestModel(modelsDir)
	isResuming := latestModel != ""

	if !isResuming {
		// Write log header only for new runs
		writeLogHeader(logFile)
	} else {
		// Log resumption
		logResumption(logFile, latestModel, startChunk)
	}

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

	fmt.Printf("📊 Full dataset sizes: Train=%d, Test=%d\n", len(trainInputs), len(testInputs))

	// Build or load network
	var nn *paragon.Network[float32]

	if isResuming {
		fmt.Printf("\n🔄 RESUMING from model: %s (starting at chunk %d)\n", latestModel, startChunk)

		// Load the latest model
		loadedNet, err := paragon.LoadNamedNetworkFromJSONFile(latestModel)
		if err != nil {
			fmt.Printf("❌ Failed to load model %s: %v\n", latestModel, err)
			fmt.Println("⚠️  Falling back to creating new network...")
			isResuming = false
		} else {
			var ok bool
			nn, ok = loadedNet.(*paragon.Network[float32])
			if !ok {
				fmt.Printf("❌ Loaded model is not float32 network\n")
				fmt.Println("⚠️  Falling back to creating new network...")
				isResuming = false
			} else {
				fmt.Println("✅ Successfully loaded saved model")
				logFile.WriteString(fmt.Sprintf("\n✅ Loaded model: %s at %s\n",
					latestModel, time.Now().Format("2006-01-02 15:04:05")))
			}
		}
	}

	if !isResuming {
		// Build initial network
		nn = paragon.NewNetwork[float32](
			[]struct{ Width, Height int }{{28, 28}, {globalMinWidth, globalMinHeight}, {10, 1}},
			[]string{"linear", "relu", "softmax"},
			[]bool{true, true, true},
		)
		startChunk = 0

		// Perform initial training on the first batch
		firstBatchInputs := trainInputs[0:initialBatchSize]
		firstBatchTargets := trainTargets[0:initialBatchSize]
		nn.Train(firstBatchInputs, firstBatchTargets, 5, learningRate, false, clippingUp, clippingLow)
		fmt.Println("🆕 Starting with fresh network")

	}

	// Enable WebGPU
	nn.WebGPUNative = true
	if err := nn.InitializeOptimizedGPU(); err != nil {
		fmt.Printf("⚠️  Failed to initialize WebGPU: %v\n", err)
		fmt.Println("   Continuing with CPU-only processing...")
		nn.WebGPUNative = false
	} else {
		fmt.Println("✅ WebGPU initialized successfully")
		defer nn.CleanupOptimizedGPU()
	}

	nn.Debug = false

	fmt.Println("🧠 Initial Network Structure:")
	printNetworkShape(nn)

	// Track models saved
	totalModelsSaved := 0

	// If resuming, get the previous total from checkpoint
	if isResuming {
		if checkpoint, err := loadCheckpoint(); err == nil {
			totalModelsSaved = checkpoint.TotalModelsSaved
			fmt.Printf("📊 Previous session saved %d models\n", totalModelsSaved)
		}
	}

	// Calculate number of chunks
	numTrainChunks := (len(trainInputs) + trainChunkSize - 1) / trainChunkSize
	numTestChunks := (len(testInputs) + testChunkSize - 1) / testChunkSize
	totalChunks := max(numTrainChunks, numTestChunks)

	fmt.Printf("\n📦 Processing dataset in %d chunks (train: %d chunks, test: %d chunks)\n",
		totalChunks, numTrainChunks, numTestChunks)

	// Process each chunk
	for chunkIdx := startChunk; chunkIdx < totalChunks; chunkIdx++ {
		fmt.Printf("\n\n🔄 ========== CHUNK %d/%d ==========\n", chunkIdx+1, totalChunks)
		logFile.WriteString(fmt.Sprintf("\n\n========== PROCESSING CHUNK %d/%d ==========\n",
			chunkIdx+1, totalChunks))
		chunkStartTime := time.Now()

		// Calculate chunk boundaries
		trainStart := chunkIdx * trainChunkSize
		trainEnd := min(trainStart+trainChunkSize, len(trainInputs))
		testStart := (chunkIdx % numTestChunks) * testChunkSize
		testEnd := min(testStart+testChunkSize, len(testInputs))

		// Skip if we've exhausted training data
		if trainStart >= len(trainInputs) {
			fmt.Printf("✅ All training data processed\n")
			break
		}

		// Extract chunk data
		chunkTrainInputs := trainInputs[trainStart:trainEnd]
		chunkTrainTargets := trainTargets[trainStart:trainEnd]
		chunkTestInputs := testInputs[testStart:testEnd]
		chunkTestTargets := testTargets[testStart:testEnd]

		fmt.Printf("📊 Chunk data - Train: [%d:%d] (%d samples), Test: [%d:%d] (%d samples)\n",
			trainStart, trainEnd, len(chunkTrainInputs),
			testStart, testEnd, len(chunkTestInputs))

		// Run growth cycle on this chunk
		session, modelsSavedInChunk := runGrowthCycle(
			nn,
			chunkTrainInputs, chunkTrainTargets,
			chunkTestInputs, chunkTestTargets,
			chunkIdx,
			trainStart, trainEnd,
			testStart, testEnd,
			logFile,
			programStartTime,
			&totalModelsSaved,
		)

		totalModelsSaved += modelsSavedInChunk

		// Log session results with timing
		chunkDuration := time.Since(chunkStartTime)
		session.Duration = chunkDuration
		logSessionResults(logFile, session)

		fmt.Printf("⏱️  Chunk %d completed in %v\n", chunkIdx, chunkDuration)

		// Save checkpoint after each chunk
		saveCheckpoint(ProcessingCheckpoint{
			LastCompletedChunk: chunkIdx,
			TotalModelsSaved:   totalModelsSaved,
			LastModelPath:      getLatestModelPath(modelsDir),
			StartTime:          programStartTime,
			LastUpdateTime:     time.Now(),
		})

		// Force garbage collection between chunks
		runtime.GC()
		debug.FreeOSMemory()
	}

	// Final evaluation on full dataset
	fmt.Println("\n\n🏁 === FINAL EVALUATION ON FULL DATASET ===")
	fmt.Println("This may take a moment...")

	fmt.Println("\n📊 Final Full Training Set Evaluation:")
	fmt.Printf("   Evaluating %d samples...\n", len(trainInputs))
	finalTrainScore := evaluateFullNetwork(nn, trainInputs, trainTargets)
	fmt.Printf("   Final Training ADHD Score: %.4f\n", finalTrainScore)

	fmt.Println("\n📊 Final Full Test Set Evaluation:")
	fmt.Printf("   Evaluating %d samples...\n", len(testInputs))
	finalTestScore := evaluateFullNetwork(nn, testInputs, testTargets)
	fmt.Printf("   Final Test ADHD Score: %.4f\n", finalTestScore)

	// Save final model
	finalModelPath := filepath.Join(modelsDir, fmt.Sprintf("mnist_model_final_train%.2f_test%.2f.json",
		finalTrainScore, finalTestScore))
	if err := nn.SaveJSON(finalModelPath); err != nil {
		fmt.Printf("❌ Failed to save final model: %v\n", err)
	} else {
		fmt.Printf("💾 Saved final model: %s\n", finalModelPath)
	}

	// Print summary
	fmt.Printf("\n📈 Summary:\n")
	fmt.Printf("   - Total chunks processed: %d\n", totalChunks-startChunk)
	fmt.Printf("   - Total models saved: %d\n", totalModelsSaved)
	fmt.Printf("   - Final train score: %.2f%%\n", finalTrainScore)
	fmt.Printf("   - Final test score: %.2f%%\n", finalTestScore)
	fmt.Printf("   - Models saved in: %s\n", modelsDir)

	// Log final summary with all models saved
	logFinalSummary(logFile, totalModelsSaved, modelsDir, startChunk, totalChunks,
		finalTrainScore, finalTestScore)

	// Clean up checkpoint file on successful completion
	if err := os.Remove(checkpointFile); err != nil && !os.IsNotExist(err) {
		fmt.Printf("⚠️  Failed to remove checkpoint file: %v\n", err)
	} else {
		fmt.Println("✅ Checkpoint file cleaned up")
	}

	// Print growth history
	printGrowthHistory(nn)

	fmt.Printf("\n📄 Growth log saved to: %s\n", logFilePath)
	fmt.Printf("🎯 Run completed successfully!\n")
}

// runGrowthCycle runs the growth attempts on a single chunk of data
func runGrowthCycle(
	nn *paragon.Network[float32],
	chunkTrainInputs, chunkTrainTargets [][][]float64,
	chunkTestInputs, chunkTestTargets [][][]float64,
	chunkIdx int,
	trainStart, trainEnd int,
	testStart, testEnd int,
	logFile *os.File,
	programStartTime time.Time,
	totalModelsSaved *int,
) (GrowthSession, int) {

	session := GrowthSession{
		ChunkIndex:    chunkIdx,
		TrainStartIdx: trainStart,
		TrainEndIdx:   trainEnd,
		TestStartIdx:  testStart,
		TestEndIdx:    testEnd,
	}

	modelsSaved := 0

	// Initial evaluation on this chunk
	fmt.Println("\n🧪 Initial Chunk Evaluation:")
	initialTrainScore := evaluateFullNetwork(nn, chunkTrainInputs, chunkTrainTargets)
	initialTestScore := evaluateFullNetwork(nn, chunkTestInputs, chunkTestTargets)
	fmt.Printf("Initial Chunk Training ADHD Score: %.4f\n", initialTrainScore)
	fmt.Printf("Initial Chunk Test ADHD Score: %.4f\n", initialTestScore)

	session.InitialScore = initialTrainScore

	// Growth parameters
	batchSize := initialBatchSize
	totalCores := runtime.NumCPU()
	maxThreads := max(1, totalCores/2)
	if maxThreads > 4 {
		maxThreads = 4
	}

	fmt.Printf("\n🌱 Starting growth attempts for chunk %d...\n", chunkIdx)
	fmt.Printf("📋 Max attempts: %d, Batch size: %d, Threads: %d\n", maxGrowthAttempts, batchSize, maxThreads)

	successfulGrowths := 0

	for attempt := 1; attempt <= maxGrowthAttempts; attempt++ {
		attemptStartTime := time.Now()
		fmt.Printf("\n🔄 === Chunk %d - Growth Attempt %d/%d ===\n", chunkIdx, attempt, maxGrowthAttempts)
		printMemoryUsage(fmt.Sprintf("chunk %d attempt %d start", chunkIdx, attempt))

		// Log attempt start
		logFile.WriteString(fmt.Sprintf("\n--- Attempt %d/%d started at %s ---\n",
			attempt, maxGrowthAttempts, attemptStartTime.Format("15:04:05")))

		// Try each possible checkpoint layer
		growthFound := false
		for checkpointLayer := 1; checkpointLayer < len(nn.Layers)-1; checkpointLayer++ {
			fmt.Printf("\n🎯 Trying checkpoint layer %d (of %d layers)\n", checkpointLayer, len(nn.Layers))

			// Prepare batch for growth attempt
			batchInputs, batchTargets := prepareBatch(chunkTrainInputs, chunkTrainTargets, batchSize, attempt)
			expectedLabels := extractLabels(batchTargets)

			// Get baseline scores
			baselineTrainScore := evaluateFullNetwork(nn, chunkTrainInputs, chunkTrainTargets)
			baselineTestScore := evaluateFullNetwork(nn, chunkTestInputs, chunkTestTargets)

			// Create a copy of the network for testing
			networkCopy, err := cloneNetwork(nn)
			if err != nil {
				fmt.Printf("❌ Failed to clone network: %v\n", err)
				continue
			}

			// Attempt to grow the network copy
			layerFound := networkCopy.Grow(
				checkpointLayer,
				batchInputs,
				expectedLabels,
				20, 3, learningRate, 1e-6,
				clippingUp, clippingLow,
				globalMinWidth, globalMaxWidth, globalMinHeight, globalMaxHeight,
				[]string{"relu", "tanh", "leaky_relu", "sigmoid", "elu"},
				maxThreads,
			)

			// Evaluate the modified copy
			var afterTrainScore, afterTestScore float64
			var newLayerInfo LayerInfo

			if layerFound {
				afterTrainScore = evaluateFullNetwork(networkCopy, chunkTrainInputs, chunkTrainTargets)
				afterTestScore = evaluateFullNetwork(networkCopy, chunkTestInputs, chunkTestTargets)
				newLayerInfo = getNewLayerInfo(nn, networkCopy)
			} else {
				afterTrainScore = baselineTrainScore
				afterTestScore = baselineTestScore
			}

			// Calculate improvement
			trainImprovement := afterTrainScore - baselineTrainScore
			testImprovement := afterTestScore - baselineTestScore

			// Accept if either improves by >1.0
			//accepted := layerFound && (trainImprovement > 0.1 || testImprovement > 0.1)
			accepted := layerFound && (trainImprovement > 0.1)
			// Log attempt
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

			// Write both standard and detailed logs
			writeGrowthAttempt(logFile, growthAttempt)
			writeDetailedGrowthAttempt(logFile, growthAttempt, nn, networkCopy, attemptStartTime)

			if accepted {
				fmt.Printf("✅ ACCEPTING GROWTH! Train: %+.4f, Test: %+.4f\n",
					trainImprovement, testImprovement)
				*nn = *networkCopy
				successfulGrowths++
				growthFound = true
				session.GrowthsAccepted++

				// Save the improved model immediately
				modelFileName := fmt.Sprintf("mnist_model_chunk%d_attempt%d_train%.2f_test%.2f.json",
					chunkIdx, attempt, afterTrainScore, afterTestScore)
				modelPath := filepath.Join(modelsDir, modelFileName)

				if err := nn.SaveJSON(modelPath); err != nil {
					fmt.Printf("❌ Failed to save improved model: %v\n", err)
				} else {
					fmt.Printf("💾 SAVED IMPROVED MODEL: %s\n", modelFileName)
					fmt.Printf("   📈 Train: %.2f%% (+%.2f), Test: %.2f%% (+%.2f)\n",
						afterTrainScore, trainImprovement, afterTestScore, testImprovement)
					modelsSaved++
					*totalModelsSaved++
				}

				break
			}
		}

		// Early termination conditions
		if successfulGrowths >= 6 {
			fmt.Printf("🎯 Reached %d successful growths, terminating early\n", successfulGrowths)
			break
		}

		// Adjust batch size if needed
		if !growthFound && attempt%5 == 0 && batchSize > 16 {
			batchSize = max(16, batchSize-8)
			fmt.Printf("🔧 Reducing batch size to %d\n", batchSize)
		}

		// Log and display progress periodically
		if attempt%5 == 0 || attempt == maxGrowthAttempts {
			elapsed := time.Since(programStartTime)
			fmt.Printf("⏰ Running for %v | Memory: %.2f MB | Models saved: %d\n",
				elapsed.Round(time.Second),
				float64(getMemoryUsage())/1024/1024,
				*totalModelsSaved)
		}

		// Garbage collection
		if attempt%3 == 0 {
			runtime.GC()
			debug.FreeOSMemory()
		}
	}

	// Final evaluation on this chunk
	finalTrainScore := evaluateFullNetwork(nn, chunkTrainInputs, chunkTrainTargets)
	finalTestScore := evaluateFullNetwork(nn, chunkTestInputs, chunkTestTargets)
	session.FinalScore = finalTrainScore

	session.ModelsSaved = modelsSaved

	fmt.Printf("\n📊 Chunk %d Final Results:\n", chunkIdx)
	fmt.Printf("   - Successful growths: %d\n", successfulGrowths)
	fmt.Printf("   - Models saved: %d\n", modelsSaved)
	fmt.Printf("   - Train score: %.2f%% → %.2f%% (%+.2f%%)\n",
		initialTrainScore, finalTrainScore, finalTrainScore-initialTrainScore)
	fmt.Printf("   - Test score: %.2f%% → %.2f%% (%+.2f%%)\n",
		initialTestScore, finalTestScore, finalTestScore-initialTestScore)

	return session, modelsSaved
}

// Helper functions

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

	clone.WebGPUNative = original.WebGPUNative
	clone.Debug = original.Debug

	return clone, nil
}

func getNewLayerInfo[T paragon.Numeric](original, modified *paragon.Network[T]) LayerInfo {
	if len(modified.Layers) > len(original.Layers) {
		for i := 1; i < len(modified.Layers)-1; i++ {
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

	lastHidden := len(modified.Layers) - 2
	return LayerInfo{
		Index:      lastHidden,
		Width:      modified.Layers[lastHidden].Width,
		Height:     modified.Layers[lastHidden].Height,
		Activation: modified.Layers[lastHidden].Neurons[0][0].Activation,
	}
}

func prepareBatch(inputs, targets [][][]float64, batchSize, attempt int) ([][][]float64, [][][]float64) {
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

func evaluateFullNetwork[T paragon.Numeric](nn *paragon.Network[T], inputs, targets [][][]float64) float64 {
	expected := make([]float64, len(inputs))
	actual := make([]float64, len(inputs))

	for i := range inputs {
		nn.Forward(inputs[i])
		out := nn.ExtractOutput()

		expected[i] = float64(paragon.ArgMax(targets[i][0]))
		actual[i] = float64(paragon.ArgMax(out))
	}

	nn.EvaluateModel(expected, actual)
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
	hostname, _ := os.Hostname()
	header := fmt.Sprintf(`
========================================
NEURAL NETWORK SYSTEMATIC GROWTH LOG
========================================
Started: %s
Host: %s
Go Version: %s
CPUs: %d
Strategy: Processing full dataset in chunks
Chunk Size: Train=%d, Test=%d
Max Attempts per chunk: %d
Models saved to: %s
Growth Strategy: Appending 32x32 layers with various activations
Acceptance Criteria: Train improvement >1.0 OR Test improvement >1.0

Log Format:
- Each attempt shows detailed metrics
- Memory usage tracked throughout
- Network architecture changes logged
- Timing information for all operations
- Model saves tracked with filenames

`, time.Now().Format("2006-01-02 15:04:05"),
		hostname,
		runtime.Version(),
		runtime.NumCPU(),
		trainChunkSize, testChunkSize, maxGrowthAttempts, modelsDir)

	file.WriteString(header)
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

func writeDetailedGrowthAttempt(file *os.File, attempt GrowthAttempt,
	originalNet, modifiedNet *paragon.Network[float32], startTime time.Time) {

	duration := time.Since(startTime)

	// Basic attempt info
	msg := fmt.Sprintf("\nGrowth Attempt #%d Details:\n", attempt.AttemptNumber)
	msg += fmt.Sprintf("├─ Timestamp: %s\n", attempt.Timestamp)
	msg += fmt.Sprintf("├─ Duration: %v\n", duration)
	msg += fmt.Sprintf("├─ Checkpoint Layer: %d\n", attempt.CheckpointLayer)
	msg += fmt.Sprintf("├─ Batch Size: %d\n", attempt.BatchSize)
	msg += fmt.Sprintf("├─ Network Layers: %d → %d\n", attempt.NetworkLayersBefore, attempt.NetworkLayersAfter)

	// Score improvements
	msg += fmt.Sprintf("├─ Scores:\n")
	msg += fmt.Sprintf("│  ├─ Train: %.4f → %.4f (improvement: %+.4f)\n",
		attempt.BeforeScore, attempt.AfterScore, attempt.Improvement)
	msg += fmt.Sprintf("│  └─ Test:  %.4f → %.4f (improvement: %+.4f)\n",
		attempt.BeforeTestScore, attempt.AfterTestScore, attempt.TestImprovement)

	// Layer details if found
	if attempt.LayerFound {
		msg += fmt.Sprintf("├─ New Layer Details:\n")
		msg += fmt.Sprintf("│  ├─ Index: %d\n", attempt.LayerIndex)
		msg += fmt.Sprintf("│  ├─ Size: %dx%d\n", attempt.LayerWidth, attempt.LayerHeight)
		msg += fmt.Sprintf("│  └─ Activation: %s\n", attempt.LayerActivation)
	}

	// Memory usage
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	msg += fmt.Sprintf("├─ Memory Usage:\n")
	msg += fmt.Sprintf("│  ├─ Allocated: %.2f MB\n", float64(m.Alloc)/1024/1024)
	msg += fmt.Sprintf("│  └─ System: %.2f MB\n", float64(m.Sys)/1024/1024)

	// Decision
	msg += fmt.Sprintf("└─ Decision: ")
	if attempt.Accepted {
		msg += "✅ ACCEPTED (Improvement threshold met)\n"
	} else if attempt.LayerFound {
		msg += "❌ REJECTED (Insufficient improvement)\n"
	} else {
		msg += "⚠️  NO LAYER FOUND\n"
	}

	// Network architecture comparison if layer was added
	if attempt.LayerFound && modifiedNet != nil {
		msg += "\nNetwork Architecture After Attempt:\n"
		for i, layer := range modifiedNet.Layers {
			marker := " "
			if i == attempt.LayerIndex {
				marker = "→"
			}
			msg += fmt.Sprintf("%s Layer %d: %dx%d (%s)\n",
				marker, i, layer.Width, layer.Height, layer.Neurons[0][0].Activation)
		}
	}

	msg += "────────────────────────────────\n"

	file.WriteString(msg)
}

func logResumption(file *os.File, modelPath string, startChunk int) {
	msg := fmt.Sprintf(`
========================================
RESUMING GROWTH SESSION
========================================
Time: %s
Loaded Model: %s
Starting from chunk: %d
========================================

`, time.Now().Format("2006-01-02 15:04:05"), modelPath, startChunk)
	file.WriteString(msg)
}

func logSessionResults(file *os.File, session GrowthSession) {
	msg := fmt.Sprintf(`
========== CHUNK %d COMPLETE ==========
Time Range: %s
Duration: %v
Train indices: [%d:%d] (%d samples)
Test indices: [%d:%d] (%d samples)
Initial score: %.4f
Final score: %.4f
Total improvement: %+.4f
Growths accepted: %d
Models saved: %d
Average time per growth: %v
Memory at completion:
`, session.ChunkIndex,
		time.Now().Format("15:04:05"),
		session.Duration,
		session.TrainStartIdx, session.TrainEndIdx,
		session.TrainEndIdx-session.TrainStartIdx,
		session.TestStartIdx, session.TestEndIdx,
		session.TestEndIdx-session.TestStartIdx,
		session.InitialScore, session.FinalScore,
		session.FinalScore-session.InitialScore,
		session.GrowthsAccepted, session.ModelsSaved,
		session.Duration/time.Duration(maxGrowthAttempts))

	// Add memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	msg += fmt.Sprintf("  - Allocated: %.2f MB\n", float64(m.Alloc)/1024/1024)
	msg += fmt.Sprintf("  - Total: %.2f MB\n", float64(m.Sys)/1024/1024)
	msg += fmt.Sprintf("  - GC runs: %d\n", m.NumGC)

	msg += "=====================================\n"

	file.WriteString(msg)
}

func logFinalSummary(file *os.File, totalModels int, modelsDir string,
	startChunk, totalChunks int, finalTrain, finalTest float64) {

	msg := fmt.Sprintf(`

========================================
FINAL SESSION SUMMARY
========================================
Completed: %s
Chunks Processed: %d (started from chunk %d)
Total Models Saved: %d
Final Scores:
  - Training: %.4f%%
  - Testing:  %.4f%%

Models Saved During This Session:
`, time.Now().Format("2006-01-02 15:04:05"),
		totalChunks-startChunk, startChunk, totalModels,
		finalTrain, finalTest)

	// List all models in directory
	files, err := os.ReadDir(modelsDir)
	if err == nil {
		modelCount := 0
		for _, file := range files {
			if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
				info, _ := file.Info()
				msg += fmt.Sprintf("  [%d] %s (%.2f MB) - %s\n",
					modelCount+1,
					file.Name(),
					float64(info.Size())/1024/1024,
					info.ModTime().Format("15:04:05"))
				modelCount++
			}
		}
	}

	msg += "\n========================================\n"
	file.WriteString(msg)
}

func printMemoryUsage(stage string) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("🧠 Memory [%s]: %.2f MB allocated, %.2f MB total\n",
		stage, float64(m.Alloc)/1024/1024, float64(m.Sys)/1024/1024)
}

// Utility functions

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func getMemoryUsage() uint64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Alloc
}

// Checkpoint functions

func findLatestModel(modelsDir string) (string, int) {
	// First check if there's a checkpoint file
	if checkpoint, err := loadCheckpoint(); err == nil {
		fmt.Printf("📌 Found checkpoint: Last completed chunk %d\n", checkpoint.LastCompletedChunk)
		return checkpoint.LastModelPath, checkpoint.LastCompletedChunk + 1
	}

	// Fallback to searching directory
	files, err := os.ReadDir(modelsDir)
	if err != nil {
		return "", 0
	}

	var latestFile string
	var latestChunk int = -1
	var latestTime time.Time

	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			// Parse chunk number from filename
			var chunk int
			if n, _ := fmt.Sscanf(file.Name(), "mnist_model_chunk%d", &chunk); n == 1 {
				info, err := file.Info()
				if err == nil && info.ModTime().After(latestTime) {
					latestTime = info.ModTime()
					latestFile = filepath.Join(modelsDir, file.Name())
					latestChunk = chunk
				}
			}
		}
	}

	if latestChunk >= 0 {
		// Return the next chunk to process
		return latestFile, latestChunk + 1
	}
	return "", 0
}

func saveCheckpoint(checkpoint ProcessingCheckpoint) error {
	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(checkpointFile, data, 0644)
}

func loadCheckpoint() (ProcessingCheckpoint, error) {
	data, err := os.ReadFile(checkpointFile)
	if err != nil {
		return ProcessingCheckpoint{}, err
	}

	var checkpoint ProcessingCheckpoint
	err = json.Unmarshal(data, &checkpoint)
	return checkpoint, err
}

func getLatestModelPath(modelsDir string) string {
	files, err := os.ReadDir(modelsDir)
	if err != nil {
		return ""
	}

	var latestFile string
	var latestTime time.Time

	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			info, err := file.Info()
			if err == nil && info.ModTime().After(latestTime) {
				latestTime = info.ModTime()
				latestFile = filepath.Join(modelsDir, file.Name())
			}
		}
	}

	return latestFile
}
