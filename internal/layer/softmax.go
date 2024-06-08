package layer

import (
	"fmt"
	"gonn/internal/mat"
	"math"
)

type SoftmaxLayer[T mat.Float] struct {
	// simple activation layer

	LayerIO[T]

	Temperature uint64
}

func NewSoftmaxLayer[T mat.Float](temp uint64) *SoftmaxLayer[T] {
	return &SoftmaxLayer[T]{
		Temperature: temp,
		LayerIO: LayerIO[T]{
			I: nil,
			O: nil,
		},
	}
}

func (sfl *SoftmaxLayer[T]) Forward(x *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	if x == nil {
		return nil, fmt.Errorf(
			"Failed to SoftmaxLayer::Forward, reason { %s }",
			"nil input provided",
		)
	}
	sfl.I = x
	sfl.O = sfl.softmax(x.Clone())

	return sfl.O, nil
}

func (sfl *SoftmaxLayer[T]) Backward(loss *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	if loss == nil {
		return nil, fmt.Errorf(
			"Failed to SoftmaxLayer::Backward, reason { %s }",
			"nil loss provided",
		)
	}

	panic("Error! SoftmaxLayer::Backward not implemented yet.")
}

func (sfl *SoftmaxLayer[T]) IsLearnable() (learnable bool, gradient *mat.Mat2D[T]) {
	return false, nil
}

func (sfl *SoftmaxLayer[T]) Learn(
	updateWeights *(func(weights, grad *mat.Mat2D[T]) (*mat.Mat2D[T], error)),
) error {
	return fmt.Errorf("Error! SoftmaxLayer is unlearnable! ")
}

func (sfl *SoftmaxLayer[T]) softmax(values *mat.Mat2D[T]) *mat.Mat2D[T] {
	temp := T(sfl.Temperature)

	sft := values.Apply(func(x T) T {
		return T(math.Exp(float64(x / temp)))
	})

	// normalize
	vsums := sft.SumRows()
	for j := range vsums.Cols() {
		vsum := vsums.MustGet(0, j)
		for i := range sft.Rows() {
			val := sft.MustGet(i, j) / vsum
			sft.MustSet(i, j, val)
		}
	}

	return sft
}
