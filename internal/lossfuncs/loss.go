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

	/*
		ce			= -((y)*log(y_) + (1-y)*log(1-y_))
		dce/dy_		= -(y * 1 / y_ + (1-y)/(1-y_)*-1)
					= -y/y_ + (1-y)/(1-y_)
	*/
	CE := mat.New2D[T](uint64(y.Rows()), uint64(y.Cols()))
	for j := range CE.Rows() {
		// jth training label

		for i := range CE.Rows() {
			Y := y.MustGet(i, j)
			Y_ := y_.MustGet(i, j)

			val := Y*T(math.Log(float64(Y_))) + (1-Y)*T(math.Log(float64(1-Y_)))

			CE.MustSet(i, j, -val)
		}
	}

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

	/*
		ce			= -((y)*log(y_) + (1-y)*log(1-y_))
		dce/dy_		= -(y * 1 / y_ + (1-y)/(1-y_)*-1)
					= -y/y_ + (1-y)/(1-y_)
	*/

	dce := mat.New2D[T](uint64(y.Rows()), uint64(y.Cols()))

	for j := range dce.Rows() {
		// jth training label

		for i := range dce.Rows() {
			Y := y.MustGet(i, j)
			Y_ := y_.MustGet(i, j)

			val := -(Y / Y_) + (1-Y)/(1-Y_)
			dce.MustSet(i, j, val)
		}
	}

	return dce, nil
}
