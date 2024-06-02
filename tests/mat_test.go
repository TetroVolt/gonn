package tests

import (
	"gonn/internal/mat"
	"testing"
)

func TestMatZeros(t *testing.T) {
	m := mat.New2DF32(2, 2)

	for i := range m.Rows() {
		for j := range m.Cols() {
			if m.At(i, j) != 0 {
				t.Errorf(
					"Expected zero value, found: %f",
					m.At(i, j),
				)
			}
		}
	}
}

func TestMatOnes(t *testing.T) {
	m := mat.Ones[float32](2, 2)

	for i := range m.Rows() {
		for j := range m.Cols() {
			if m.At(i, j) != 1.0 {
				t.Errorf(
					"Expected zero value, found: %f",
					m.At(i, j),
				)
			}
		}
	}
}

func TestMatARange(t *testing.T) {
	m := mat.ARange[float32](4)
	if val := m.At(0, 0); val != 0.0 {
		t.Errorf("Expected value of 0.0, found: %f", val)
	}
	if val := m.At(0, 1); val != 1.0 {
		t.Errorf("Expected value of 1.0, found: %f", val)
	}
	if val := m.At(0, 2); val != 2.0 {
		t.Errorf("Expected value of 2.0, found: %f", val)
	}
	if val := m.At(0, 3); val != 3.0 {
		t.Errorf("Expected value of 3.0, found: %f", val)
	}
}

func TestMatReshape(t *testing.T) {
	m1 := mat.ARange[float32](6)
	m2, err := m1.Reshape(3, 2)
	if err != nil {
		t.Fatal(err)
	}

	if m2.Rows() != 3 || m2.Cols() != 2 {
		t.Fatalf(
			"Expected shape [3,2], found: [%d, %d]",
			m2.Rows(), m2.Cols(),
		)
	}

	if val := m2.At(0, 0); val != 0.0 {
		t.Errorf("Expected value of 0.0, found: %f", val)
	}
	if val := m2.At(0, 1); val != 1.0 {
		t.Errorf("Expected value of 1.0, found: %f", val)
	}
	if val := m2.At(1, 0); val != 2.0 {
		t.Errorf("Expected value of 2.0, found: %f", val)
	}
	if val := m2.At(1, 1); val != 3.0 {
		t.Errorf("Expected value of 3.0, found: %f", val)
	}
	if val := m2.At(2, 0); val != 4.0 {
		t.Errorf("Expected value of 4.0, found: %f", val)
	}
	if val := m2.At(2, 1); val != 5.0 {
		t.Errorf("Expected value of 5.0, found: %f", val)
	}

	m3, err := m2.Reshape(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if m3.Rows() != 2 || m3.Cols() != 3 {
		t.Fatalf(
			"Expected shape [2,3], found: [%d, %d]",
			m3.Rows(), m3.Cols(),
		)
	}
	if val := m3.At(0, 0); val != 0.0 {
		t.Errorf("Expected value of 0.0, found: %f", val)
	}
	if val := m3.At(0, 1); val != 1.0 {
		t.Errorf("Expected value of 1.0, found: %f", val)
	}
	if val := m3.At(0, 2); val != 2.0 {
		t.Errorf("Expected value of 2.0, found: %f", val)
	}
	if val := m3.At(1, 0); val != 3.0 {
		t.Errorf("Expected value of 3.0, found: %f", val)
	}
	if val := m3.At(1, 1); val != 4.0 {
		t.Errorf("Expected value of 4.0, found: %f", val)
	}
	if val := m3.At(1, 2); val != 5.0 {
		t.Errorf("Expected value of 5.0, found: %f", val)
	}
}

func TestMatSetAt(t *testing.T) {
	m := mat.New2DF32(2, 2)
	m.Set(0, 0, 1.0)
	m.Set(0, 1, 2.0)
	m.Set(1, 0, 3.0)
	m.Set(1, 1, 4.0)

	one := m.At(0, 0)
	two := m.At(0, 1)
	three := m.At(1, 0)
	four := m.At(1, 1)

	if one != 1.0 ||
		two != 2.0 ||
		three != 3.0 ||
		four != 4.0 {

		t.Errorf(
			"Expected values to be: [1,2,3,4], found [%f, %f, %f, %f]",
			one, two, three, four,
		)
	}
}

func TestMatEquality(t *testing.T) {
	m1 := mat.ARange[float32](4).MustReshape(2, 2)
	m2 := mat.ARange[float32](4).MustReshape(2, 2)
	m3 := mat.ARange[float32](4)
	m4 := mat.Ones[float32](2, 2)

	if m1.At(0, 0) != m2.At(0, 0) ||
		m1.At(0, 1) != m2.At(0, 1) ||
		m1.At(1, 0) != m2.At(1, 0) ||
		m1.At(1, 1) != m2.At(1, 1) {
		t.Fatal("Mismatched m1 and m2 using At")
	}

	if !mat.Equals(m1, m2) {
		t.Error("Expected m1 == m2")
	}

	if mat.Equals(m2, m3) {
		t.Error("Expected m2 != m3")
	}

	if mat.Equals(m4, m1) || mat.Equals(m4, m2) {
		t.Error("Expected m4 != m1 and m4 != m2")
	}
}

func TestMatAdd(t *testing.T) {
	m1 := mat.New2DF32(2, 2)
	m2 := mat.Ones[float32](2, 2)

	m3 := mat.New2DF32(2, 2)
	m3, err := mat.MatAdd(m3, m1, m2)

	if err != nil {
		t.Fatalf("Error adding: %v\n", err)
	}

	if m2.At(0, 0) != m3.At(0, 0) ||
		m2.At(0, 1) != m3.At(0, 1) ||
		m2.At(1, 0) != m3.At(1, 0) ||
		m2.At(1, 1) != m3.At(1, 1) {
		t.Errorf("Expected m3 and m2 to both be Ones matrix")
	}
}

func TestMatMatMul(t *testing.T) {
	A := mat.ARange[float32](4).MustReshape(2, 2)
	B := mat.ARange[float32](4).MustReshape(2, 2)

	C, err := mat.MatMul(A, B)
	if err != nil {
		t.Fatalf("Failed MatMul test: %s", err)
	}

	/*
		[0 1] [0 1] = [0*0+1*2  0*1+1*3] = [2  3]
		[2 3] [2 3] = [2*0+3*2  2*1+3*3] = [6 11]
	*/

	if val := C.At(0, 0); val != 2 {
		t.Errorf("Expected m[0,0]=2 found: %f", val)
	}
	if val := C.At(0, 1); val != 3 {
		t.Errorf("Expected m[0,1]=3 found: %f", val)
	}
	if val := C.At(1, 0); val != 6 {
		t.Errorf("Expected m[1,0]=6 found: %f", val)
	}
	if val := C.At(1, 1); val != 11 {
		t.Errorf("Expected m[1,1]=11 found: %f", val)
	}
}
