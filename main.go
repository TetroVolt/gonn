package main

import (
	"fmt"
	"gonn/internal/mat"
	"log"

	"math/rand"
)

func main() {
	matDemo()
}

func simplePerceptron() {
	// Train a perceptron to do XOR

	trainData := mat.FromValues([]float32{
		// XOR truth table
		0.0, 0.0, 0.0,
		1.0, 0.0, 1.0,
		0.0, 1.0, 1.0,
		1.0, 1.0, 0.0,
	})

	X := trainData.MustSlice(0, 0, 4, 2)
	y := trainData.MustSlice(0, 2, 4, 1)

	inputWeights := mat.FromValues(randF32Slice(2*3)).MustReshape(2, 3)

	// appease lsp for now
	_, _, _ = X, y, inputWeights
}

func randF32Slice(count uint64) []float32 {
	slice := make([]float32, count)
	for i := range count {
		slice[i] = rand.Float32()
	}
	return slice
}

func matDemo() {
	fmt.Println("Gonn")

	m1 := mat.Ones[float32](2, 2)
	m2 := mat.Ones[float32](2, 4)

	fmt.Println(m1.String())
	fmt.Println(m1.MustStringify())

	fmt.Println(m2.String())
	fmt.Println(m2.MustStringify())

	m3, err := mat.Mul(m1, m2)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(m3.String())
	fmt.Println(m3.MustStringify())
}
