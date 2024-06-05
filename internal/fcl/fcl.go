package fcl

import (
	"fmt"
	"gonn/internal/mat"
)

type FullyConnectedLayer struct {
	// Store intermediate results for easier backprop
	I  *mat.Mat2DF32 // Input
	W  *mat.Mat2DF32 // weights
	Wx *mat.Mat2DF32 // Weighted Input
	O  *mat.Mat2DF32 // Activated Weighted Input, aka. Output

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
		O:  nil,

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
	if fcl.O == nil {
		return nil, fmt.Errorf(
			"Cannot get FCL Output because output is nil",
		)
	}
	return fcl.O, nil
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

	O := Wx.Clone().Apply(ac)

	fcl.I = I
	fcl.Wx = Wx
	fcl.O = O

	return O, nil
}

func (fcl *FCL) Backward(loss *mat.Mat2DF32) error {
	panic("TODO Not implemented yet. ")

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
