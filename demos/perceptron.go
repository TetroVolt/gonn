package demos

import (
	"fmt"
	"math"
	"math/rand"

	"gonn/internal/mat"
)

func PerceptronDemo() {
	// Train a perceptron to do XOR
	activate := func(v *mat.Mat2DF32) (*mat.Mat2DF32, error) {
		sigged := v.Clone()
		for i := range sigged.Rows() {
			for j := range sigged.Cols() {
				val, err := sigged.Get(i, j)
				if err != nil {
					return nil, err
				}

				if err := sigged.Set(i, j, sigmoid(val)); err != nil {
					return nil, err
				}
			}
		}
		return sigged, nil
	}

	forward := func(x *mat.Mat2DF32, w *mat.Mat2DF32) (*mat.Mat2DF32, error) {
		ones := mat.Ones[float32](1, uint64(x.Cols()))
		xwb, err := mat.VCat(ones, x) // xwb ~ X with Bias (ones)
		if err != nil {
			return nil, err
		}

		WX, err := mat.Mul(w, xwb)
		if err != nil {
			return nil, err
		}

		AWX, err := activate(WX)
		if err != nil {
			return nil, err
		}

		return AWX, nil
	}

	trainData := mat.FromValues([]float32{
		// XOR truth table
		0.0, 0.0, 0.0,
		1.0, 0.0, 1.0,
		0.0, 1.0, 1.0,
		1.0, 1.0, 0.0,
	}).MustReshape(4, 3)

	X := trainData.MustSlice(mat.SR{0, 4}, mat.SR{0, 2})
	y := trainData.MustSlice(mat.SR{0, 4}, mat.SR{2, 3})

	W1 :=
		mat.FromValues(
			randF32Slice(2*3),
		).MustReshape(2, 3)

	W2 :=
		mat.FromValues(
			randF32Slice(2*3),
		).MustReshape(2, 3)

	W3 :=
		mat.FromValues(
			randF32Slice(1*3),
		).MustReshape(1, 3)

	fmt.Println("Perceptron Demo (XOR)")
	fmt.Printf("X:\n%s\n", X.MustStringify())
	fmt.Printf("y:\n%s\n", y.MustStringify())

	res1, err := forward(X.TP(), W1)
	if err != nil {
		fmt.Printf(
			"Error forwarding X for W1.\n%s\n%s\n%s\n",
			fmt.Sprintf("X:\n%s\n", X.MustStringify()),
			fmt.Sprintf("W1:\n%s\n", W1.MustStringify()),
			err,
		)
		return
	}

	res2, err := forward(res1, W2)
	if err != nil {
		fmt.Printf(
			"Error forwarding res1 for W1.\n%s\n%s\n%s\n",
			fmt.Sprintf("res1:\n%s\n", res1.MustStringify()),
			fmt.Sprintf("W2:\n%s\n", W2.MustStringify()),
			err,
		)

		return
	}

	y_, err := forward(res1, W3)
	if err != nil {
		fmt.Printf(
			"Error forwarding res2 for W3.\n%s\n%s\n%s\n",
			fmt.Sprintf("res2:\n%s\n", res2.MustStringify()),
			fmt.Sprintf("W3:\n%s\n", W3.MustStringify()),
			err,
		)

		return
	}

	yy_, err := mat.HCat(X, y, y_.TP())
	if err != nil {
		fmt.Printf(
			"Error! Failed to HCat [X , y , y_].\n%s\n%s\n%s",
			fmt.Sprintf(
				"X:\n%s\n",
				X.MustStringify(),
			),
			fmt.Sprintf(
				"y:\n%s\n",
				y.MustStringify(),
			),
			fmt.Sprintf(
				"y_:\n%s\n",
				y_.MustStringify(),
			),
		)
		return
	}

	fmt.Println("[   X  ,   y   ,  y_   ]")
	fmt.Printf("%s\n", yy_.MustStringify())
}

func randF32Slice(count uint64) []float32 {
	slice := make([]float32, count)
	for i := range count {
		slice[i] = rand.Float32()
	}
	return slice
}

func sigmoid(x float32) float32 {
	exp := float32(math.Exp(float64(-x)))
	return 1 / (1 + exp)
}

func dsigmoid(x float32) float32 {
	sv := sigmoid(x)
	return sv * (1 - sv)
}
