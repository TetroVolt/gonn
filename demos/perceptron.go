package demos

import (
	"fmt"
	"log"
	"slices"

	"gonn/internal/acti"
	"gonn/internal/layer"
	"gonn/internal/lossfuncs"
	"gonn/internal/mat"
)

func PerceptronDemo() {
	trainData := mat.FromValues([]float32{
		// XOR truth table
		0.0, 0.0, 0.0,
		1.0, 0.0, 1.0,
		0.0, 1.0, 1.0,
		1.0, 1.0, 0.0,
	}).MustReshape(4, 3)

	X := trainData.MustSlice(
		mat.RS{0, 4}, mat.CS{0, 2},
	).TP()
	y := trainData.MustSlice(
		mat.RS{0, 4}, mat.CS{2, 3},
	).TP()

	modelLayers := createModel()

	fmt.Println("Perceptron Demo (XOR)")
	fmt.Printf("X:\n%s\n", X.MustStringify())
	fmt.Printf("y:\n%s\n", y.MustStringify())

	const ALPHA = 0.1
	for range 100000 {
		y_, err := forward(modelLayers, X)
		if err != nil {
			log.Fatalf("Failed to forward model, reason = { %s }", err)
		}

		se, err := lossfuncs.SquaredError(y, y_)
		if err != nil {
			log.Fatalf("Failed to get SquaredError, reason { %s }", err)
		}
		fmt.Printf("MeanSquareError: %f\n", se.Sum()/float32(y_.Cols()))

		dse, err := lossfuncs.DSquaredError(y, y_)
		if err != nil {
			log.Fatalf("Failed to get DSquaredError, reason { %s }", err)
		}
		_, err = backward(modelLayers, dse)
		if err != nil {
			log.Fatalf("Failed to get BackProp, reason { %s }", err)
		}

		err = updateWeights(modelLayers, ALPHA)
		if err != nil {
			log.Fatalf("Failed to update weights, reason { %s }", err)
		}
	}

	y_, err := forward(modelLayers, X)
	if err != nil {
		log.Fatalf("Failed to forward model, reason = { %s }", err)
	}

	stats(X, y, y_)
	ce, err := lossfuncs.CrossEntropy(y, y_)
	if err != nil {
		log.Fatalf("Failed to get CrossEntropy, reason { %s }", err)
	}
	fmt.Printf("MeanCrossEntropy: %f\n", ce.Sum()/float32(ce.Cols()))
}

func updateWeights(model []layer.Layer[float32], alpha float32) error {
	if model == nil || len(model) == 0 {
		return fmt.Errorf("nil or zero length model provided")
	}

	weightUpdater := func(weights, grad *mat.Mat2DF32) (*mat.Mat2DF32, error) {
		err := weights.Subtract(grad.Scale(alpha))
		if err != nil {
			return nil, err
		}
		return weights, nil
	}

	for _, layer := range model {
		learnable, _ := layer.IsLearnable()
		if !learnable {
			continue
		}

		err := layer.Learn(&weightUpdater)
		if err != nil {
			return err
		}
	}

	return nil
}

func backward(model []layer.Layer[float32], L *mat.Mat2DF32) ([]*mat.Mat2DF32, error) {
	if model == nil || len(model) == 0 {
		return nil, fmt.Errorf("nil or zero length model provided")
	}
	if L == nil || L.Cols() == 0 {
		return nil, fmt.Errorf("L is nil or zero length")
	}

	var loss *mat.Mat2DF32 = L
	gradients := make([](*mat.Mat2DF32), len(model))

	for i := len(model) - 1; i >= 0; i-- {
		layer := model[i]

		grad, err := layer.Backward(loss)
		if err != nil {
			return nil, fmt.Errorf(
				"model backprop failed at layer[%d of %d], reason: { %s }",
				i+1, len(model), err,
			)
		}

		gradients[i] = grad
		loss = grad
	}

	slices.Reverse(gradients)
	return gradients, nil
}

func forward(model []layer.Layer[float32], X *mat.Mat2DF32) (*mat.Mat2DF32, error) {
	if model == nil || len(model) == 0 {
		return nil, fmt.Errorf("nil or zero length model provided")
	}
	if X == nil || X.Cols() == 0 {
		return nil, fmt.Errorf("X is nil or zero length")
	}

	var inp, out *mat.Mat2DF32 = X, nil
	var err error

	for i, layer := range model {
		out, err = layer.Forward(inp)
		if err != nil {
			return nil, fmt.Errorf(
				"model forwarding failed at layer[%d], reason: { %s }",
				i, err,
			)
		}
		inp = out
	}

	return out, nil
}

func createModel() []layer.Layer[float32] {
	// Instantiate Activation Functions
	sigmoid := acti.NewAF[float32](acti.Sigmoid)
	dSigmoid := acti.NewAF[float32](acti.DSigmoid)

	// Instantiate Layers
	modelLayers := []layer.Layer[float32]{
		layer.NewLL[float32](2, 2),
		layer.NewAL(
			sigmoid,
			dSigmoid,
		),
		layer.NewLL[float32](2, 2),
		layer.NewAL(
			sigmoid,
			dSigmoid,
		),
		layer.NewLL[float32](2, 1),
		layer.NewAL(
			sigmoid,
			dSigmoid,
		),
	}

	return modelLayers
}

func stats(X, y, y_ *mat.Mat2DF32) {
	if X == nil || y == nil || y_ == nil {
		log.Fatalf(
			"Failed to print stats, nil parameters provided for "+
				"stats(X == nil : %t , y == nil : %t , y_ == nil : %t)",
			X == nil,
			y == nil,
			y_ == nil,
		)
		return
	}

	yy_, err := mat.HCat(X.TP(), y.TP(), y_.TP())
	if err != nil {
		fmt.Printf(
			"Error! Failed to HCat [X , y , y_]."+
				"\n%s\n%s\n%s",

			fmt.Sprintf(
				"X:\n%s\n",
				X.MustStringify(),
			),
			fmt.Sprintf(
				"y:\n%s\n",
				y.MustStringify(),
			),
			fmt.Sprintf(
				"y_:\n%s\n",
				y_.MustStringify(),
			),
		)
		return
	}

	fmt.Println("[   X  ,   y   ,  y_   ]")
	fmt.Printf("%s\n", yy_.MustStringify())
}
