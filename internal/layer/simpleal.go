package layer

import (
	"fmt"
	"gonn/internal/mat"
)

type SimpleAL[T mat.Float] struct {
	// simple activation layer

	AF  *(func(X T) T) // activation function
	DAF *(func(X T) T) // derivative of activation function
	LayerIO[T]
}

func NewAL[T mat.Float](AF, DAF *(func(X T) T)) *SimpleAL[T] {
	return &SimpleAL[T]{
		AF:  AF,
		DAF: DAF,

		LayerIO: LayerIO[T]{
			I: nil,
			O: nil,
		},
	}
}

func (al *SimpleAL[T]) Forward(x *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	if x == nil {
		return nil, fmt.Errorf(
			"Failed to SimpleAL::Forward, reason { %s }",
			"nil input provided",
		)
	}
	al.I = x
	al.O = x.Clone().Apply(*al.AF)

	return al.O, nil
}

func (al *SimpleAL[T]) Backward(loss *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	if loss == nil {
		return nil, fmt.Errorf(
			"Failed to SimpleAL::Backward, reason { %s }",
			"nil loss provided",
		)
	}

	back, err := mat.Mul(loss, al.I.Clone().Apply(*al.DAF))
	if err != nil {
		return nil, fmt.Errorf(
			"Failed to SimpleAL::Backward, reason { %s }",
			err,
		)
	}

	return back, nil
}

func (al *SimpleAL[T]) IsLearnable() (learnable bool, gradient *mat.Mat2D[T]) {
	return false, nil
}

func (al *SimpleAL[T]) Learn(
	updateWeights *(func(weights, grad *mat.Mat2D[T]) (*mat.Mat2D[T], error)),
) error {
	return fmt.Errorf("Error! SimpleAL is unlearnable! ")
}
