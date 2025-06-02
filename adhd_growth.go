package main

import (
	"fmt"
	"time"

	"github.com/openfluke/paragon"
)

func growNetworkADHD[T paragon.Numeric](nn *paragon.Network[T], inputs, targets [][][]float64, iterations int) {
	fmt.Println("\n🌱 Starting ADHD-guided growth for MNIST...")

	config := paragon.DefaultGrowConfig()
	config.BatchSize = 64
	config.MicroNetCount = 160
	config.MinADHDScore = 70.0
	config.ImprovementThreshold = 0.1
	config.TrainingEpochs = 0
	config.LearningRate = 0.01
	config.NewLayerWidth = 16
	config.MaxGrowthAttempts = 200
	config.Debug = false

	initialScore := nn.EvaluateFullScore(inputs, targets)
	fmt.Printf("📊 Initial ADHD Score: %.2f\n", initialScore)

	for i := 1; i <= iterations; i++ {
		fmt.Printf("\n🔥 Growth Iteration %d/%d\n", i, iterations)
		start := time.Now()
		result, err := nn.Grow(inputs, targets, config)
		duration := time.Since(start)

		if err != nil {
			fmt.Printf("❌ Growth error: %v\n", err)
			break
		}
		if result.Success {
			fmt.Printf("✅ Success: ADHD %.2f → %.2f (+%.2f) in %v\n",
				result.OriginalADHDScore, result.ImprovedADHDScore, result.ADHDImprovement, duration)
		} else {
			fmt.Println("⚠️  No improvement this round.")
		}
	}
}
