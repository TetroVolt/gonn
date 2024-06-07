package lossfuncs

import (
	"fmt"
	"gonn/internal/mat"
	"math"
)

func SquaredError[T mat.Float](y, y_ *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	se, err := mat.Subtract(y, y_) // y - y_
	if err != nil {
		return nil, fmt.Errorf("Failed to SE, reason { %s }", err)
	}

	se.Apply(func(x T) T { return x * x }) // (y - y_) ^ 2
	se.Scale(0.5)                          // (1/2) * ((y - y_) ^ 2)

	return se, nil
}

func DSquaredError[T mat.Float](y, y_ *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	mse, err := mat.Subtract(y_, y)
	if err != nil {
		return nil, err
	}
	return mse, nil
}

func CrossEntropy[T mat.Float](y, y_ *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	if !mat.DimsMatch(y, y_) {
		return nil, fmt.Errorf(
			"Cannot perform CrossEntropy loss on matrices with mismatched dims: "+
				"y[%d, %d]\ty_[%d, %d]",
			y.Rows(), y.Cols(),
			y_.Rows(), y_.Cols(),
		)
	}

	logX := y_.Clone().Apply(
		func(x T) T {
			return T(math.Log(float64(x)))
		},
	)

	log1_X := y_.Clone().Apply(
		func(x T) T {
			return T(math.Log(float64(1 - x)))
		},
	)

	CEY, err := mat.Mul(y, logX)
	if err != nil {
		return nil, err
	}

	CE1_Y, err := mat.Mul(
		y.Clone().Apply(func(x T) T { return 1 - x }),
		log1_X,
	)
	if err != nil {
		return nil, err
	}

	CE, err := mat.Add(CEY, CE1_Y)
	if err != nil {
		return nil, err
	}
	CE.Scale(-1)

	return CE, nil
}

func DCrossEntropy[T mat.Float](y, y_ *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	if !mat.DimsMatch(y, y_) {
		return nil, fmt.Errorf(
			"Cannot perform DCrossEntropy loss on matrices with mismatched dims: "+
				"y[%d, %d]\ty_[%d, %d]",
			y.Rows(), y.Cols(),
			y_.Rows(), y_.Cols(),
		)
	}

	panic("DCrossEntropy not implemented yet.")
}
