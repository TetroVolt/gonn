package layer

import (
	"gonn/internal/mat"
)

type LayerIO[T mat.Float] struct {
	I *mat.Mat2D[T] // input
	O *mat.Mat2D[T] // output
}

type Propagatable[T mat.Float] interface {
	Forward(x *mat.Mat2D[T]) (*mat.Mat2D[T], error)
	Backward(loss *mat.Mat2D[T]) (*mat.Mat2D[T], error)
}

type Learnable[T mat.Float] interface {
	IsLearnable() (learnable bool, gradient *mat.Mat2D[T])
	Learn(
		updateWeights *(func(weights, grad *mat.Mat2D[T]) (*mat.Mat2D[T], error)),
	) error
}

type Layer[T mat.Float] interface {
	Propagatable[T]
	Learnable[T]
}
