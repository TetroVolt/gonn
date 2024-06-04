package demos

import (
	"fmt"
	"gonn/internal/mat"
	"log"
)

func MatDemo() {
	fmt.Println("Mat Demo")

	m1 := mat.Ones[float32](2, 2)
	m2 := mat.Ones[float32](2, 4)

	fmt.Println(m1.String())
	fmt.Println(m1.MustStringify())

	fmt.Println(m2.String())
	fmt.Println(m2.MustStringify())

	m3, err := mat.Mul(m1, m2)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(m3.String())
	fmt.Println(m3.MustStringify())
}
