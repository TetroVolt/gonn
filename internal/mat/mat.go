package mat

import (
	"fmt"
	"log"
	"math/rand"
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

// Aliases
type Mat2DF32 = Mat2D[float32]
type Mat2DF64 = Mat2D[float64]

// Constructors
func New2D[T Float](rows, cols uint64) *Mat2D[T] {
	return &Mat2D[T]{
		rows:   rows,
		cols:   cols,
		stride: cols,

		transposed: false,
		values:     make([]T, rows*cols),
	}
}

func New2DF32(rows, cols uint64) *Mat2DF32 {
	return New2D[float32](rows, cols)
}

func New2DF64(rows, cols uint64) *Mat2DF64 {
	return New2D[float64](rows, cols)
}

func FromValues[T Float](values []T) *Mat2D[T] {
	N := uint64(len(values))

	return &Mat2D[T]{
		rows:   1,
		cols:   N,
		stride: N,

		transposed: false,
		values:     values,
	}
}

func (m *Mat2D[T]) TP() *Mat2D[T] {
	return m.Transpose()
}

func (m *Mat2D[T]) Transpose() *Mat2D[T] {
	return &Mat2D[T]{
		transposed: !m.transposed,
		rows:       m.cols,
		cols:       m.rows,
		stride:     m.stride,

		values: m.values,
	}
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

type SliceRange = [2]int64
type SR = SliceRange
type RS = SliceRange // Row Slice alias
type CS = SliceRange // Col Slice alias

type MatSlice struct {
	R SliceRange
	C SliceRange
}

func (m *Mat2D[T]) Slice(rs, cs SliceRange) (*Mat2D[T], error) {
	rsl, csl, err := m.validateMS(rs, cs)

	if err != nil {
		return nil, err
	}

	i, j := uint64(rsl[0]), uint64(csl[0])
	rows, cols := uint64(rsl[1])-i, uint64(csl[1])-j

	slicedMat, err := m.slice(i, j, rows, cols)
	if err != nil {
		return nil, err
	}

	return slicedMat, nil
}

func (m *Mat2D[T]) MustSlice(rs, cs SliceRange) *Mat2D[T] {
	sl, err := m.Slice(rs, cs)
	if err != nil {
		log.Fatal(err)
	}
	return sl
}

func (m *Mat2D[T]) Clone() *Mat2D[T] {
	rows, cols := uint64(m.Rows()), uint64(m.Cols())
	clone := New2D[T](rows, cols)

	for i := range clone.Rows() {
		for j := range clone.Cols() {
			clone.MustSet(i, j, m.MustGet(i, j))
		}
	}

	return clone
}

func (m *Mat2D[T]) Apply(f func(T) T) *Mat2D[T] {
	for i := range m.Rows() {
		for j := range m.Cols() {
			val := f(m.MustGet(i, j))
			m.MustSet(i, j, val)
		}
	}
	return m
}

func (m *Mat2D[T]) Fill(val T) *Mat2D[T] {
	for i := range m.Rows() {
		for j := range m.Cols() {
			m.MustSet(i, j, val)
		}
	}
	return m
}

func Ones[T Float](rows, cols uint64) *Mat2D[T] {
	mat := New2D[T](rows, cols)
	mat.Fill(T(1.0))
	return mat
}

func Id[T Float](length uint64) *Mat2D[T] { // identity matrix
	mat := New2D[T](length, length)

	for i := range int64(length) {
		mat.MustSet(i, i, 1.0)
	}

	return mat
}

func Rand[T Float](rows, cols uint64) *Mat2D[T] {
	mat := New2D[T](rows, cols)
	mat.Apply(func(x T) T {
		return T(rand.Float64())
	})
	return mat
}

func RandF32(rows, cols uint64) *Mat2DF32 {
	mat := New2DF32(rows, cols)
	for i := range len(mat.values) {
		mat.values[i] = rand.Float32()
	}
	return mat
}

func RandF64(rows, cols uint64) *Mat2DF64 {
	mat := New2DF64(rows, cols)
	for i := range len(mat.values) {
		mat.values[i] = rand.Float64()
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
		m,
	)
}

func (m *Mat2D[T]) Scale(sc T) *Mat2D[T] {
	for i := range m.Rows() {
		for j := range m.Cols() {
			m.MustSet(i, j, (m.MustGet(i, j) * sc))
		}
	}

	return m
}

func (a *Mat2D[T]) MustAdd(b *Mat2D[T]) *Mat2D[T] {
	if err := add(a, a, b); err != nil {
		log.Fatal(err)
	}
	return a
}

func (a *Mat2D[T]) Add(b *Mat2D[T]) error {
	if err := add(a, a, b); err != nil {
		return err
	}
	return nil
}

func (a *Mat2D[T]) MustSubtract(b *Mat2D[T]) *Mat2D[T] {
	if err := subtract(a, a, b); err != nil {
		log.Fatal(err)
	}
	return a
}

func (a *Mat2D[T]) Subtract(b *Mat2D[T]) error {
	if err := subtract(a, a, b); err != nil {
		return err
	}
	return nil
}

func (a *Mat2D[T]) MustMul(b *Mat2D[T]) *Mat2D[T] {
	if err := subtract(a, a, b); err != nil {
		log.Fatal(err)
	}
	return a
}

func (a *Mat2D[T]) Mul(b *Mat2D[T]) error {
	if err := mul(a, a, b); err != nil {
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

func Subtract[T Float](a, b *Mat2D[T]) (*Mat2D[T], error) {
	dst := New2D[T](a.rows, a.cols)
	if err := subtract(dst, a, b); err != nil {
		return nil, err
	}
	return dst, nil
}

func Mul[T Float](a, b *Mat2D[T]) (*Mat2D[T], error) {
	dst := New2D[T](a.rows, a.cols)
	if err := mul(dst, a, b); err != nil {
		return nil, err
	}
	return dst, nil
}

func MustAdd[T Float](a, b *Mat2D[T]) *Mat2D[T] {
	a, err := Subtract(a, b)
	if err != nil {
		log.Fatal(err)
	}
	return a
}

func MustSubtract[T Float](a, b *Mat2D[T]) *Mat2D[T] {
	difference, err := Subtract(a, b)
	if err != nil {
		log.Fatal(err)
	}
	return difference
}

func MustMul[T Float](a, b *Mat2D[T]) *Mat2D[T] {
	dot, err := Mul(a, b)
	if err != nil {
		log.Fatal(err)
	}
	return dot
}

func MatMul[T Float](a, b *Mat2D[T]) (*Mat2D[T], error) {
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

func VCat[T Float](matrices ...(*Mat2D[T])) (*Mat2D[T], error) {
	rows, cols, err := dimsCanVCat(matrices...)
	if err != nil {
		return nil, err
	}

	res := New2D[T](rows, cols)

	currRow := int64(0)
	for _, mat := range matrices {

		// Copy the matrix
		// TODO figure out if copy(dst, src) is faster
		//   -> need to figure out how to deal with transposed

		for i := range mat.Rows() {
			for j := range mat.Cols() {
				value := mat.MustGet(i, j)
				res.MustSet(currRow+i, j, value)
			}
		}

		currRow += mat.Rows()
	}

	return res, nil
}

func HCat[T Float](matrices ...(*Mat2D[T])) (*Mat2D[T], error) {
	rows, cols, err := dimsCanHCat(matrices...)
	if err != nil {
		return nil, err
	}

	res := New2D[T](rows, cols)

	currCol := int64(0)
	for _, mat := range matrices {

		// Copy the matrix
		for i := range mat.Rows() {
			for j := range mat.Cols() {
				// TODO figure out if copy(dst, src) is faster
				//   -> need to figure out how to deal with transposed

				value := mat.MustGet(i, j)
				res.MustSet(i, currCol+j, value)
			}
		}

		currCol += mat.Cols()
	}

	return res, nil
}

/*
* Sum Rows
*
* Given an M by N matrix mat[M, N],
* returns a new 1 by N matrix S[1, N] which is vector representing the sum of row vectors of mat
**/
func (mat *Mat2D[T]) SumRows() *Mat2D[T] {
	sum := New2D[T](1, mat.cols)
	for j := range mat.Cols() {
		for i := range mat.Rows() {
			val := sum.MustGet(0, j) + mat.MustGet(i, j)
			sum.MustSet(0, j, val)
		}
	}
	return sum
}

/*
* Sum Cols
* Given an M by N matrix mat[M, N],
* returns an M by 1 matrix S[M, 1] which is the vector representing the sum of the column vectors of mat
**/
func (mat *Mat2D[T]) SumCols() *Mat2D[T] {
	sum := New2D[T](mat.rows, 1)
	for i := range mat.Rows() {
		for j := range mat.Cols() {
			val := sum.MustGet(i, 0) + mat.MustGet(i, j)
			sum.MustSet(i, 0, val)
		}
	}
	return sum
}

func (mat *Mat2D[T]) Sum() T {
	var sum T = 0
	for i := range mat.Rows() {
		for j := range mat.Cols() {
			sum += mat.MustGet(i, j)
		}
	}
	return sum
}

// vvv PRIVATE vvv

func (m *Mat2D[T]) validateEndSlice(rEnd, cEnd int64) (uint64, uint64, error) {
	var rE, cE uint64

	if rEnd == m.Rows() {
		rE = m.rows
	} else if -m.Rows() < rEnd && rEnd < m.Rows() {
		rE = uint64(
			(((rEnd % m.Rows()) + m.Rows()) % m.Rows()),
		)
	} else {
		return 0, 0, fmt.Errorf(
			"Invalid endSlice[:%d,:%d] for M%s, end row value %d out of bounds. ",
			rEnd, cEnd,
			m.stringifyRowCol(),
			rEnd,
		)
	}

	if cEnd == m.Cols() {
		cE = m.cols
	} else if -m.Cols() < cEnd && cEnd < m.Cols() {
		cE = uint64(
			(((cEnd % m.Cols()) + m.Cols()) % m.Cols()),
		)
	} else {
		return 0, 0, fmt.Errorf(
			"Invalid endSlice[:%d,:%d] for M%s, end col value %d out of bounds. ",
			rEnd, cEnd,
			m.stringifyRowCol(),
			cEnd,
		)
	}

	return rE, cE, nil
}

func (m *Mat2D[T]) validateMS(R, C SR) (rSlice SR, cSlice SR, err error) {
	rStart, rEnd := R[0], R[1]
	cStart, cEnd := C[0], C[1]

	rS, cS, err := m.posIndexes(rStart, cStart)
	if err != nil {
		return SR{}, SR{}, err
	}

	rE, cE, err := m.validateEndSlice(rEnd, cEnd)
	if err != nil {
		return SR{}, SR{}, fmt.Errorf(
			"Invalid slice[%v, %v] for M%s, reason: %s",
			R, C, m.stringifyRowCol(),
			err,
		)
	}

	if !(rS < rE && cS < cE) {
		return SR{}, SR{}, fmt.Errorf(
			"Invalid Slice [%v, %v] for %s", R, C, m.stringifyRowCol(),
		)
	}

	rSlice = SR{int64(rS), int64(rE)}
	cSlice = SR{int64(cS), int64(cE)}
	err = nil
	return
}

func dimsCanHCat[T Float](matrices ...(*Mat2D[T])) (rows, cols uint64, err error) {
	cols = 0
	rows = matrices[0].rows

	for _, mat := range matrices {
		if mat.Rows() != matrices[0].Rows() {
			return 0, 0, fmt.Errorf(
				"Cannot horizontally cat matrix, mismatched columns: %s, %s",
				matrices[0].stringifyRowCol(),
				mat.stringifyRowCol(),
			)
		} else {
			cols += uint64(mat.Cols())
		}
	}

	return rows, cols, nil
}

func dimsCanVCat[T Float](matrices ...(*Mat2D[T])) (rows, cols uint64, err error) {
	rows = 0
	cols = matrices[0].cols

	for _, mat := range matrices {
		if mat.Cols() != matrices[0].Cols() {
			return 0, 0, fmt.Errorf(
				"Cannot vertically cat matrix, mismatched columns: %s, %s",
				matrices[0].stringifyRowCol(),
				mat.stringifyRowCol(),
			)
		} else {
			rows += uint64(mat.Rows())
		}
	}

	return rows, cols, nil
}

func (m *Mat2D[T]) stringifyRowCol() string {
	return fmt.Sprintf("[%d, %d]", m.Rows(), m.Cols())
}

func (m *Mat2D[T]) posIndexes(i, j int64) (uint64, uint64, error) {
	rows, cols := m.Rows(), m.Cols()

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
			"Error! Mismatched rows or cols %s != %s",
			a.stringifyRowCol(),
			b.stringifyRowCol(),
		)
		return err
	}
	return nil
}

func dimsCanMul[T Float](a, b *Mat2D[T]) error {
	if b.rows != a.cols {
		return fmt.Errorf(
			"Error! Invalid dims for AB mat mult: A%s B%s",
			a.stringifyRowCol(),
			b.stringifyRowCol(),
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

func subtract[T Float](dst, a, b *Mat2D[T]) error {
	if err := validateDimsMatch(dst, a, b); err != nil {
		return err
	}

	rows, cols := a.Rows(), a.Cols()

	for i := range rows {
		for j := range cols {
			dst.Set(i, j, (a.MustGet(i, j) - b.MustGet(i, j)))
		}
	}

	return nil
}

func add[T Float](dst, a, b *Mat2D[T]) error {
	if err := validateDimsMatch(dst, a, b); err != nil {
		return err
	}

	rows, cols := a.Rows(), a.Cols()

	for i := range rows {
		for j := range cols {
			dst.Set(i, j, (a.MustGet(i, j) + b.MustGet(i, j)))
		}
	}

	return nil
}

func mul[T Float](dst, a, b *Mat2D[T]) error {
	if err := validateDimsMatch(dst, a, b); err != nil {
		return err
	}

	rows, cols := a.Rows(), a.Cols()

	for i := range rows {
		for j := range cols {
			dst.Set(i, j, (a.MustGet(i, j) * b.MustGet(i, j)))
		}
	}

	return nil
}

func validateDimsMatch[T Float](dst, a, b *Mat2D[T]) error {
	if !(DimsMatch(a, b) && DimsMatch(dst, a)) {
		return fmt.Errorf(
			"Mismatched dims dst%s = a%s - b%s",
			dst.stringifyRowCol(),
			a.stringifyRowCol(),
			b.stringifyRowCol(),
		)
	}

	return nil
}

func (m *Mat2D[T]) slice(i, j, r, c uint64) (*Mat2D[T], error) {
	// i and j determine start location
	// r and c are how much to take

	if !(i+r <= m.rows && j+c <= m.cols) {
		err := fmt.Errorf(
			"Invalid matrix slice dims ->"+
				"m[%d, %d], s[%d:%d, %d:%d]",
			m.rows, m.cols,
			i, i+r, j, j+c,
		)

		return nil, err
	}

	// startIndex := (i)*(m).stride + (j)
	startIndex, err := m.valueIndex(int64(i), int64(j))
	if err != nil {
		return nil, err
	}

	submat := Mat2D[T]{
		rows:   r,
		cols:   c,
		stride: m.stride,

		transposed: m.transposed,
		values:     m.values[startIndex:],
	}

	return &submat, nil
}
