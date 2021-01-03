package training

import "github.com/Mas281/gonet/mat"

type Data struct {
	Input  []float64
	Target []float64
}

func (d Data) TargetMatrix() mat.Matrix {
	return mat.NewColumn(d.Target)
}
