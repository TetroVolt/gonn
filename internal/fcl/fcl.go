package fcl

import (
	"fmt"
	"gonn/internal/mat"
	"math"
)

type FullyConnectedLayer struct {
	// Store intermediate results for easier backprop
	I  *mat.Mat2DF32 // Input
	W  *mat.Mat2DF32 // weights
	Wx *mat.Mat2DF32 // Weighted Input
	Ac *mat.Mat2DF32 // Activated Weighted Input, aka. Output

	iSize, oSize uint64
}
type FCL = FullyConnectedLayer

func NewFCL(iSize, oSize uint64) *FCL {
	cols := iSize + 1 // first column for the biases
	rows := oSize
	W := mat.RandF32(rows, cols)

	fcl := FCL{
		I:  nil,
		W:  W,
		Wx: nil,
		Ac: nil,

		iSize: iSize, oSize: oSize,
	}

	return &fcl
}

func (fcl *FCL) GetISize() uint64 {
	return fcl.iSize
}

func (fcl *FCL) GetOSize() uint64 {
	return fcl.oSize
}

func (fcl *FCL) GetOutput() (*mat.Mat2DF32, error) {
	if fcl.Ac == nil {
		return nil, fmt.Errorf(
			"Cannot get FCL Output because output is nil",
		)
	}
	return fcl.Ac, nil
}

func (fcl *FCL) Forward(
	x *mat.Mat2DF32, ac func(float32) float32,
) (*mat.Mat2DF32, error) {

	I, err := fcl.prepInput(x)
	if err != nil {
		return nil, err
	}

	Wx, err := mat.Mul(fcl.W, I)
	if err != nil {
		return nil, err
	}

	Ac := Wx.Clone().Apply(ac)

	fcl.I = I
	fcl.Wx = Wx
	fcl.Ac = Ac

	return Ac, nil
}

func (fcl *FCL) Backward(loss *mat.Mat2DF32) error {
	panic("Not implemented yet. ")
}

func Sigmoid(x float32) float32 {
	exp := float32(math.Exp(float64(-x)))
	return 1 / (1 + exp)
}

func DSigmoid(x float32) float32 {
	sv := Sigmoid(x)
	return sv * (1 - sv)
}

// ## private ##

func (fcl *FCL) prepInput(X *mat.Mat2DF32) (*mat.Mat2DF32, error) {
	if X.Rows()+1 != fcl.W.Cols() {
		return nil, fmt.Errorf(
			"Invalid input shape for W[%d, 1 + %d], found X[1 + %d, %d]",
			fcl.W.Rows(), fcl.W.Cols()-1,
			X.Rows(), X.Cols(),
		)
	}

	I, err := mat.VCat(
		mat.Ones[float32](1, uint64(X.Cols())),
		X,
	)

	if err != nil {
		return nil, err
	}

	return I, nil
}
