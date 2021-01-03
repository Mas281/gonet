package mat

import (
	"errors"
	"strconv"
)

type Matrix [][]float64

var errInvalidSize = errors.New("number of rows and columns must be positive")

func NewEmpty(rows, columns int) Matrix {
	if rows < 1 || columns < 1 {
		panic(errInvalidSize)
	}

	return zeroes(rows, columns)
}

var errWrongNoValues = errors.New("wrong number of values")

func FromSlice(rows, columns int, values []float64) Matrix {
	if len(values) != rows*columns {
		panic(errWrongNoValues)
	}

	m := NewEmpty(rows, columns)
	t := 0

	for r := 0; r < rows; r++ {
		for c := 0; c < columns; c++ {
			m.Set(c, r, values[t])
			t++
		}
	}

	return m
}

func NewColumn(values []float64) Matrix {
	data := make([][]float64, len(values))

	for i := range data {
		data[i] = []float64{values[i]}
	}

	return data
}

var errOutOfBounds = errors.New("index is out of bonds")

func (m Matrix) Dims() (int, int) {
	return m.Rows(), m.Columns()
}

func (m Matrix) Rows() int {
	return len(m)
}

func (m Matrix) Columns() int {
	return len(m[0])
}

func SameDims(a, b Matrix) bool {
	return a.Rows() == b.Rows() && a.Columns() == b.Columns()
}

func (m Matrix) Get(x, y int) float64 {
	if m.outOfBonds(x, y) {
		panic(errOutOfBounds)
	}

	return m[y][x]
}

func (m Matrix) Set(x, y int, value float64) {
	if m.outOfBonds(x, y) {
		panic(errOutOfBounds)
	}

	m[y][x] = value
}

func (m Matrix) outOfBonds(x, y int) bool {
	return x > m.Columns()-1 ||
		y > m.Rows()-1
}

func (m Matrix) Scale(scale float64) {
	m.Apply(func(num float64) float64 {
		return scale * num
	})
}

func (m Matrix) Apply(mapping func(float64) float64) {
	for _, row := range m {
		for i, element := range row {
			row[i] = mapping(element)
		}
	}
}

var errAdditionSize = errors.New("dimensions do not match for addition")

func Add(a, b Matrix) Matrix {
	if !SameDims(a, b) {
		panic(errAdditionSize)
	}

	return add(a, b, false)
}

var errSubtractionSize = errors.New("dimensions do not match for subtraction")

func Subtract(a, b Matrix) Matrix {
	if !SameDims(a, b) {
		panic(errSubtractionSize)
	}

	return add(a, b, true)
}

func add(a, b Matrix, sub bool) Matrix {
	result := NewEmpty(a.Dims())

	for r := 0; r < a.Rows(); r++ {
		for c := 0; c < a.Columns(); c++ {
			var value float64

			if sub {
				value = a.Get(c, r) - b.Get(c, r)
			} else {
				value = a.Get(c, r) + b.Get(c, r)
			}

			result.Set(c, r, value)
		}
	}

	return result
}

var errDotProductSize = errors.New("dimensions do not match for dot product")

func Dot(a, b Matrix) Matrix {
	if a.Columns() != b.Rows() {
		panic(errDotProductSize)
	}

	result := NewEmpty(a.Rows(), b.Columns())

	for r := 0; r < a.Rows(); r++ {
		for c := 0; c < b.Columns(); c++ {
			sum := 0.0

			for k := 0; k < a.Columns(); k++ {
				sum += a.Get(k, r) * b.Get(c, k)
			}

			result.Set(c, r, sum)
		}
	}

	return result
}

var errMulElemSize = errors.New("dimensions do not match for element-wise multiplication")

func MulElem(a, b Matrix) Matrix {
	if !SameDims(a, b) {
		panic(errMulElemSize)
	}

	result := NewEmpty(a.Dims())

	for r := 0; r < a.Rows(); r++ {
		for c := 0; c < a.Columns(); c++ {
			result.Set(c, r, a.Get(c, r)*b.Get(c, r))
		}
	}

	return result
}

func Transpose(m Matrix) Matrix {
	data := make([][]float64, m.Columns())

	for c := 0; c < m.Columns(); c++ {
		column := make([]float64, m.Rows())

		for r := 0; r < m.Rows(); r++ {
			column[r] = m.Get(c, r)
		}

		data[c] = column
	}

	return data
}

func Copy(m Matrix) Matrix {
	source := [][]float64(m)
	duplicate := make([][]float64, len(source))

	for i := range source {
		duplicate[i] = make([]float64, len(source[i]))
		copy(duplicate[i], source[i])
	}

	return duplicate
}

func zeroes(rows, columns int) Matrix {
	data := make([][]float64, rows)

	for i := range data {
		data[i] = make([]float64, columns)
	}

	return data
}

func (m Matrix) String() string {
	s := "["

	for i, row := range m {
		if i != 0 {
			s += " "
		}

		s += sliceToString(row)

		if i != len(m)-1 && len(m) > 1 {
			s += "\n"
		}
	}

	return s + "]"
}

func sliceToString(values []float64) string {
	s := "["

	for i, element := range values {
		s += strconv.FormatFloat(element, 'f', 3, 64)

		if i != len(values)-1 {
			s += ", "
		}
	}

	return s + "]"
}
