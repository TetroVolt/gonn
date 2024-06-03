package mat

import (
	"fmt"
	"log"
	"strings"
)

type Float interface {
	float32 | float64
}

type Mat2D[T Float] struct {
	rows   uint64
	cols   uint64
	stride uint64

	values []T

	transposed bool
}

type Mat2DF32 = Mat2D[float32]
type Mat2DF64 = Mat2D[float64]

// Constructors
func New2D[T Float](rows, cols uint64) *Mat2D[T] {
	// TODO handle zero rows or cols
	return &Mat2D[T]{
		rows:   rows,
		cols:   cols,
		stride: cols,

		values: make([]T, rows*cols),
	}
}

func New2DF32(rows, cols uint64) *Mat2DF32 {
	return New2D[float32](rows, cols)
}

func New2DF64(rows, cols uint64) *Mat2DF64 {
	return New2D[float64](rows, cols)
}

// End Constructors

func (m *Mat2D[T]) Rows() int64 {
	return int64(m.rows)
}
func (m *Mat2D[T]) Cols() int64 {
	return int64(m.cols)
}

func (m *Mat2D[T]) Get(i, j int64) (T, error) {
	index, err := m.valueIndex(i, j)
	if err != nil {
		return 0, err
	}
	return (m.values[index]), nil
}

func (m *Mat2D[T]) MustGet(i, j int64) T {
	value, err := m.Get(i, j)
	if err != nil {
		log.Fatal(err)
	}
	return value
}

func (m *Mat2D[T]) Set(i, j int64, val T) error {
	index, err := m.valueIndex(i, j)
	if err != nil {
		return err
	}
	m.values[index] = val
	return err
}

func (m *Mat2D[T]) MustSet(i, j int64, val T) {
	if err := m.Set(i, j, val); err != nil {
		log.Fatal(err)
	}
}

func FromValues[T Float](values []T) *Mat2D[T] {
	N := uint64(len(values))

	return &Mat2D[T]{
		rows:   1,
		cols:   N,
		stride: N,

		values: values,
	}
}

func (m *Mat2D[T]) SliceMat(i, j, r, c uint64) (*Mat2D[T], error) {
	if !(i < r && j < c && r <= m.rows && c <= m.cols) {
		err := fmt.Errorf(
			"Invalid matrix slice dims ->"+
				"m[%d, %d], s[%d:%d, %d:%d]",
			m.rows, m.cols,
			i, i+r, j, j+c,
		)

		return nil, err
	}

	startIndex := (i)*(m).stride + (j)
	submat := Mat2D[T]{
		rows:   r,
		cols:   c,
		stride: m.stride,

		values: m.values[startIndex:],
	}

	return &submat, nil
}

func (m *Mat2D[T]) Clone() *Mat2D[T] {
	clone := Mat2D[T]{
		rows:   m.rows,
		cols:   m.cols,
		stride: m.stride,

		values: make([]T, len(m.values)),
	}

	for i := range len(m.values) {
		clone.values[i] = m.values[i]
	}

	return &clone
}

func Ones[T Float](rows, cols uint64) *Mat2D[T] {
	mat := New2D[T](rows, cols)

	for i := range len(mat.values) {
		mat.values[i] = 1.0
	}

	return mat
}

func ARange[T Float](upto uint64) *Mat2D[T] {
	mat := New2D[T](1, upto)

	for i := range upto {
		mat.values[i] = T(i)
	}

	return mat
}

func (m *Mat2D[T]) Reshape(rows, cols uint64) (*Mat2D[T], error) {
	if err := m.mustBeReshapeable(rows, cols); err != nil {
		return nil, err
	}

	m.rows = rows
	m.cols = cols
	m.stride = cols

	return m, nil
}

func (m *Mat2D[T]) MustReshape(rows, cols uint64) *Mat2D[T] {
	res, err := m.Reshape(rows, cols)
	if err != nil {
		log.Fatal(err)
	}
	return res
}

func (m *Mat2D[T]) Stringify() (string, error) {
	var builder strings.Builder

	builder.WriteString("[\n")
	for i := range m.Rows() {
		builder.WriteString("\t[")

		for j := range m.Cols() {
			val, err := m.Get(i, j)
			if err != nil {
				return "", err
			}

			_, err = builder.WriteString(
				fmt.Sprintf("%f, ", val),
			)
			if err != nil {
				return "", err
			}
		}

		builder.WriteString("],\n")
	}
	builder.WriteString("]\n")

	return builder.String(), nil
}

func (m *Mat2D[T]) MustStringify() string {
	s, err := m.Stringify()
	if err != nil {
		log.Fatal(err)
	}
	return s
}

func (m *Mat2D[T]) String() string {
	return fmt.Sprintf(
		"%#v",
		&Mat2D[T]{
			rows: m.rows,
			cols: m.cols,
		},
	)
}

func (m *Mat2D[T]) PrintMat() {
	s, err := m.Stringify()
	if err != nil {
		log.Fatal(
			fmt.Sprintf(
				"Error! Unable to stringify matrix: %s", err))
	}
	fmt.Println(s)
}

func (m *Mat2D[T]) Scale(sc T) {
	for i := range m.Rows() {
		for j := range m.Cols() {
			m.Set(i, j, (m.MustGet(i, j) * sc))
		}
	}
}

func (a *Mat2D[T]) Add(b *Mat2D[T]) error {
	if err := add(a, a, b); err != nil {
		return err
	}
	return nil
}

func Equals[T Float](a, b *Mat2D[T]) bool {
	if !DimsMatch(a, b) {
		return false
	}

	for i := range a.Rows() {
		for j := range b.Cols() {
			if a.MustGet(i, j) != b.MustGet(i, j) {
				return false
			}
		}
	}

	return true
}

func Add[T Float](a, b *Mat2D[T]) (*Mat2D[T], error) {
	dst := New2D[T](a.rows, a.cols)
	if err := add(dst, a, b); err != nil {
		return nil, err
	}
	return dst, nil
}

func Mul[T Float](a, b *Mat2D[T]) (*Mat2D[T], error) {
	if err := dimsCanMul(a, b); err != nil {
		return nil, err
	}

	// a.cols == b.rows
	res := New2D[T](a.rows, b.cols)
	for i := range res.Rows() {
		for j := range res.Cols() {
			var sum T = 0
			for k := range b.Rows() {
				sum += a.MustGet(i, k) * b.MustGet(k, j)
			}
			res.Set(i, j, sum)
		}
	}

	return res, nil
}

func DimsMatch[T Float](a, b *Mat2D[T]) bool {
	return a.rows == b.rows && a.cols == b.cols
}

// vvv PRIVATE vvv

func (m *Mat2D[T]) posIndexes(i, j int64) (uint64, uint64, error) {
	rows := m.Rows()
	cols := m.Cols()

	if i < -rows || i >= rows || j < -cols || j >= cols {
		// index out of bounds
		return 0, 0, fmt.Errorf(
			"Index[%d, %d] out of bounds[%d, %d]",
			i, j, rows, cols,
		)
	}

	ii := uint64(((i % rows) + rows) % rows)
	jj := uint64(((j % cols) + cols) % cols)

	return ii, jj, nil
}

func (m *Mat2D[T]) valueIndex(i, j int64) (uint64, error) {
	ii, jj, err := m.posIndexes(i, j)
	if err != nil {
		return 0, err
	}
	if m.transposed {
		ii, jj = jj, ii
	}
	index := (ii)*(m).stride + (jj)
	return index, nil
}

func matchDims[T Float](a, b *Mat2D[T]) error {
	if !DimsMatch(a, b) {
		err := fmt.Errorf(
			"Error! Mismatched rows or cols [%d, %d]!=[%d, %d]",
			a.rows, a.cols,
			b.rows, b.cols,
		)
		return err
	}
	return nil
}

func dimsCanMul[T Float](a, b *Mat2D[T]) error {
	if b.rows != a.cols {
		return fmt.Errorf(
			"Error! Invalid dims for AB mat mult: A[%d, %d] B[%d, %d]",
			a.rows, a.cols,
			b.rows, b.cols,
		)
	}
	return nil
}

func (m *Mat2D[T]) reshapable(rows, cols uint64) bool {
	return m.rows*m.cols == rows*cols
}

func (m *Mat2D[T]) mustBeReshapeable(rows, cols uint64) error {
	if !m.reshapable(rows, cols) {
		err := fmt.Errorf(
			"Invalid reshape[%d, %d] for M[%d, %d]",
			rows, cols, m.rows, m.cols,
		)
		return err
	}
	return nil
}

func add[T Float](dst, a, b *Mat2D[T]) error {
	if !(DimsMatch(a, b) && DimsMatch(dst, a)) {
		return fmt.Errorf(
			"Mismatched dims dst[%d, %d] = a[%d, %d] + b[%d, %d]",
			dst.rows, dst.cols,
			a.rows, a.cols,
			b.rows, b.cols,
		)
	}

	for i := range a.Rows() {
		for j := range a.Cols() {
			dst.Set(i, j, (a.MustGet(i, j) + b.MustGet(i, j)))
		}
	}

	return nil
}
