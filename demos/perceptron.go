package demos

import (
	"fmt"
	"gonn/internal/acti"
	"gonn/internal/fcl"
	"gonn/internal/mat"
	"log"
)

func PerceptronDemo() {
	trainData := mat.FromValues([]float32{
		// XOR truth table
		0.0, 0.0, 0.0,
		1.0, 0.0, 1.0,
		0.0, 1.0, 1.0,
		1.0, 1.0, 0.0,
	}).MustReshape(4, 3)

	X := trainData.MustSlice(
		mat.RS{0, 4}, mat.CS{0, 2},
	)
	y := trainData.MustSlice(
		mat.RS{0, 4}, mat.CS{2, 3},
	)

	fmt.Println("Perceptron Demo (XOR)")
	fmt.Printf("X:\n%s\n", X.MustStringify())
	fmt.Printf("y:\n%s\n", y.MustStringify())

	L1 := fcl.NewFCL(2, 2)
	L2 := fcl.NewFCL(2, 2)
	L3 := fcl.NewFCL(2, 1)

	O1, err := L1.Forward(X.TP(), acti.Sigmoid)
	if err != nil {
		log.Fatal(err)
	}

	O2, err := L2.Forward(O1, acti.Sigmoid)
	if err != nil {
		log.Fatal(err)
	}

	y_, err := L3.Forward(O2, acti.Sigmoid)

	yy_, err := mat.HCat(X, y, y_.TP())
	if err != nil {
		fmt.Printf(
			"Error! Failed to HCat [X , y , y_]."+
				"\n%s\n%s\n%s",

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

	// fmt.Printf("L1:\n%v\n\n", L1)
	// fmt.Printf("L2:\n%v\n\n", L2)
	// fmt.Printf("L3:\n%v\n\n", L3)
}
