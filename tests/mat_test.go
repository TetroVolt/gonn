package tests

import (
	"fmt"
	"gonn/internal/mat"
	"testing"
)

func TestMatZeros(t *testing.T) {
	m := mat.New2DF32(2, 2)

	for i := range m.Rows() {
		for j := range m.Cols() {
			if m.MustGet(i, j) != 0 {
				t.Errorf(
					"Expected zero value, found: %f",
					m.MustGet(i, j),
				)
			}
		}
	}
}

func TestMatOnes(t *testing.T) {
	m := mat.Ones[float32](2, 2)

	for i := range m.Rows() {
		for j := range m.Cols() {
			if m.MustGet(i, j) != 1.0 {
				t.Errorf(
					"Expected zero value, found: %f",
					m.MustGet(i, j),
				)
			}
		}
	}
}

func TestMatARange(t *testing.T) {
	m := mat.ARange[float32](4)
	for i := range int64(4) {
		expected := float32(i)
		if val := m.MustGet(0, i); val != expected {
			t.Errorf("Expected value of %f, found: %f", expected, val)
		}
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

	logIfErr(t, expectValueAt(m2, 0, 0, 0.0))
	logIfErr(t, expectValueAt(m2, 0, 1, 1.0))
	logIfErr(t, expectValueAt(m2, 1, 0, 2.0))
	logIfErr(t, expectValueAt(m2, 1, 1, 3.0))
	logIfErr(t, expectValueAt(m2, 2, 0, 4.0))
	logIfErr(t, expectValueAt(m2, 2, 1, 5.0))

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

	logIfErr(t, expectValueAt(m3, 0, 0, 0.0))
	logIfErr(t, expectValueAt(m3, 0, 1, 1.0))
	logIfErr(t, expectValueAt(m3, 0, 2, 2.0))
	logIfErr(t, expectValueAt(m3, 1, 0, 3.0))
	logIfErr(t, expectValueAt(m3, 1, 1, 4.0))
	logIfErr(t, expectValueAt(m3, 1, 2, 5.0))
}

func TestMatSetGet(t *testing.T) {
	m := mat.New2DF32(2, 2)
	m.Set(0, 0, 1.0)
	m.Set(0, 1, 2.0)
	m.Set(1, 0, 3.0)
	m.Set(1, 1, 4.0)

	logIfErr(t, expectValueAt(m, 0, 0, 1.0))
	logIfErr(t, expectValueAt(m, 0, 1, 2.0))
	logIfErr(t, expectValueAt(m, 1, 0, 3.0))
	logIfErr(t, expectValueAt(m, 1, 1, 4.0))

	// negative indices
	m.Set(0, 0, 2.0)
	m.Set(0, -1, 3.0)
	m.Set(-1, 0, 4.0)
	m.Set(-1, -1, 5.0)

	logIfErr(t, expectValueAt(m, 0, 0, 2.0))
	logIfErr(t, expectValueAt(m, 0, -1, 3.0))
	logIfErr(t, expectValueAt(m, -1, 0, 4.0))
	logIfErr(t, expectValueAt(m, -1, -1, 5.0))
}

func TestMatEquality(t *testing.T) {
	m1 := mat.ARange[float32](4).MustReshape(2, 2)
	m2 := mat.ARange[float32](4).MustReshape(2, 2)
	m3 := mat.ARange[float32](4)
	m4 := mat.Ones[float32](2, 2)

	if m1.MustGet(0, 0) != m2.MustGet(0, 0) ||
		m1.MustGet(0, 1) != m2.MustGet(0, 1) ||
		m1.MustGet(1, 0) != m2.MustGet(1, 0) ||
		m1.MustGet(1, 1) != m2.MustGet(1, 1) {
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
	m3, err := mat.Add(m1, m2)

	if err != nil {
		t.Fatalf("Error adding: %v\n", err)
	}

	logIfErr(t, expectValueAt(m3, 0, 0, 1.0))
	logIfErr(t, expectValueAt(m3, 0, 1, 1.0))
	logIfErr(t, expectValueAt(m3, 1, 0, 1.0))
	logIfErr(t, expectValueAt(m3, 1, 1, 1.0))

	if err := m3.Add(m3); err != nil {
		t.Fatal(err)
	}
	logIfErr(t, expectValueAt(m3, 0, 0, 2.0))
	logIfErr(t, expectValueAt(m3, 0, 1, 2.0))
	logIfErr(t, expectValueAt(m3, 1, 0, 2.0))
	logIfErr(t, expectValueAt(m3, 1, 1, 2.0))

	if err := m3.Add(m3); err != nil {
		t.Fatal(err)
	}
	logIfErr(t, expectValueAt(m3, 0, 0, 4.0))
	logIfErr(t, expectValueAt(m3, 0, 1, 4.0))
	logIfErr(t, expectValueAt(m3, 1, 0, 4.0))
	logIfErr(t, expectValueAt(m3, 1, 1, 4.0))

	m4 := mat.New2DF32(2, 1)
	if err := m3.Add(m4); err == nil {
		t.Error("Expected error when adding matrices with mismatched dims, none found")
	}
}

func TestMatMul(t *testing.T) {
	A := mat.ARange[float32](4).MustReshape(2, 2)
	B := mat.ARange[float32](4).MustReshape(2, 2)

	C, err := mat.Mul(A, B)
	if err != nil {
		t.Fatalf("Failed MatMul test: %s", err)
	}

	/*
		Expected behavior:
		[0 1] [0 1] = [0*0+1*2  0*1+1*3] = [2  3]
		[2 3] [2 3] = [2*0+3*2  2*1+3*3] = [6 11]
	*/
	logIfErr(t, expectValueAt(C, 0, 0, 2))
	logIfErr(t, expectValueAt(C, 0, 1, 3))
	logIfErr(t, expectValueAt(C, 1, 0, 6))
	logIfErr(t, expectValueAt(C, 1, 1, 11))
}

func logIfErr(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Error(err)
	}
}

func expectValueAt[T mat.Float](m *mat.Mat2D[T], i, j int64, expected T) error {
	found, err := m.Get(i, j)

	if err != nil || expected != found {
		return fmt.Errorf(
			"Expected value of %f at [%d, %d], found %f",
			expected,
			i, j,
			found,
		)
	}

	return nil
}
