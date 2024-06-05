package acti

import (
	"gonn/internal/mat"
	"math"
)

func Linear[T mat.Float](x T) T {
	return x
}

func RELU[T mat.Float](x T) T {
	if x > 0 {
		return x
	}
	return 0
}

func DRELU[T mat.Float](x T) T {
	if x > 0 {
		return 1
	}
	return 0
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
