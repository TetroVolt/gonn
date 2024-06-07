package acti

import (
	"gonn/internal/mat"
	"math"
)

func NewAF[T mat.Float](
	af func(x T) T,
) *(func(x T) T) {
	f := &af
	return f
}

func Linear[T mat.Float](x T) T {
	return x
}

func ReLU[T mat.Float](x T) T {
	if x > 0 {
		return x
	}
	return 0
}

func DReLU[T mat.Float](x T) T {
	if x > 0 {
		return 1
	}
	return 0
}

func LReLU[T mat.Float](x T) T { // Leaky RELU
	if x > 0 {
		return x
	}
	return 0.01 * x
}

func DLReLU[T mat.Float](x T) T { // Leaky RELU
	if x > 0 {
		return 1
	}
	return -0.01
}

func Sigmoid[T mat.Float](x T) T {
	exp := math.Exp(float64(-x))
	sig := 1 / (1 + exp)
	return T(sig)
}

func DSigmoid[T mat.Float](x T) T {
	sm := Sigmoid(x)
	return sm * (1 - sm)
}

func SoftPlus[T mat.Float](x T) T {
	sp := math.Log(1 + float64(x))
	return T(sp)
}

func DSoftPlus[T mat.Float](x T) T {
	return Sigmoid(x)
}
