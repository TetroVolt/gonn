package layer

import (
	"fmt"
	"gonn/internal/mat"
)

type LinearLayer[T mat.Float] struct {
	LayerIO[T]

	W            *mat.Mat2D[T] // weights
	WGrad        *mat.Mat2D[T] // weights
	iSize, oSize uint64
}

func NewLL[T mat.Float](iSize, oSize uint64) *LinearLayer[T] {
	wrows, wcols := oSize, 1+iSize // for bias

	W := mat.Rand[T](wrows, wcols).Apply(func(x T) T { return x*2 - 1 })

	ll := LinearLayer[T]{
		W:     W,
		WGrad: nil,

		iSize: iSize,
		oSize: oSize,

		LayerIO: LayerIO[T]{
			I: nil,
			O: nil,
		},
	}

	return &ll
}

func (ll *LinearLayer[T]) ISize() int64 {
	return int64(ll.iSize)
}

func (ll *LinearLayer[T]) OSize() int64 {
	return int64(ll.iSize)
}

func (ll *LinearLayer[T]) Forward(x *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	if x == nil {
		return nil, ll.wrapForwardErr(fmt.Errorf("nil input"))
	}

	I, err := ll.prepForwardInput(x)
	if err != nil {
		return nil, ll.wrapForwardErr(err)
	}

	O, err := mat.MatMul(ll.W, I)
	if err != nil {
		return nil, ll.wrapForwardErr(err)
	}

	ll.I = I
	ll.O = O

	return O, nil
}

func (ll *LinearLayer[T]) Backward(loss *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	if loss == nil {
		return nil, ll.wrapForwardErr(fmt.Errorf("nil loss"))
	}

	/*
		dL/dO = loss[oSize, N]

		O[oSize, N] = W[oSize, 1 + iSize] * I[1 + iSize, N]

		dL/dW	= dL/dO * dO/dW
				= loss[oSize, N] * d/dW (WI)
				= loss[oSize, N] * I^T[N, 1 + iSize]
				= (dL/dW)[oSize, 1 + iSize]

		dL/dI   = W^T[1 + iSize, oSize] * loss[oSize, N]
				= (dL/dW)[1 + iSize, N]
	*/

	Wgrad, err := mat.MatMul(loss, ll.I.TP())
	if err != nil {
		return nil, ll.wrapBackwardErr(err)
	}
	ll.WGrad = Wgrad

	back, err := mat.MatMul(ll.W.TP(), loss)
	if err != nil {
		return nil, ll.wrapBackwardErr(err)
	}

	// Chop off bias
	back, err = back.Slice(mat.RS{1, back.Rows()}, mat.CS{0, back.Cols()})
	if err != nil {
		return nil, ll.wrapBackwardErr(err)
	}

	return back, nil
}

func (ll *LinearLayer[T]) IsLearnable() (learnable bool, gradient *mat.Mat2D[T]) {
	return true, ll.WGrad
}

func (ll *LinearLayer[T]) Learn(
	updateWeights *(func(weights, grad *mat.Mat2D[T]) (*mat.Mat2D[T], error)),
) error {
	if ll.WGrad == nil {
		return fmt.Errorf("linearlayer is learnable but gradient is nil")
	}

	newWeights, err := (*updateWeights)(
		ll.W.Clone(),
		ll.WGrad.Clone(),
	) // Cloning is expensive :(

	if err != nil {
		return fmt.Errorf("Learning Error occured getting newWeights, reason = { %s }", err)
	}
	if newWeights == nil {
		return fmt.Errorf("Learning Error updating weights, newWeights are nil")
	}

	if !mat.DimsMatch(ll.W, newWeights) {
		return fmt.Errorf(
			"Learning Error updating weights, newWeights[%d, %d] does not match dimensions of oldWeights[%d, %d]",
			newWeights.Rows(), newWeights.Cols(),
			ll.W.Rows(), ll.W.Cols(),
		)
	}

	ll.W = newWeights

	return nil
}

// ## private ##
func (ll *LinearLayer[T]) wrapForwardErr(err error) error {
	return fmt.Errorf(
		"Failed to %s::Forward, reason: { %s }",
		ll.shapeRep(),
		err,
	)
}

func (ll *LinearLayer[T]) wrapBackwardErr(err error) error {
	return fmt.Errorf(
		"Failed to %s::Backward, reason: { %s }",
		ll.shapeRep(),
		err,
	)
}

func (ll *LinearLayer[T]) shapeRep() string {
	return fmt.Sprintf(
		"<LinearLayer([%d, N]) -> [%d, N]>",
		ll.iSize, ll.oSize,
	)
}

func (ll *LinearLayer[T]) prepForwardInput(X *mat.Mat2D[T]) (*mat.Mat2D[T], error) {
	if X.Rows()+1 != ll.W.Cols() {
		return nil, fmt.Errorf(
			"Invalid input shape for W[%d, 1 + %d], found X[1 + %d, %d]",
			ll.W.Rows(), ll.W.Cols()-1,
			X.Rows(), X.Cols(),
		)
	}

	I, err := mat.VCat(
		mat.Ones[T](1, uint64(X.Cols())),
		X,
	)

	if err != nil {
		return nil, err
	}

	return I, nil
}
