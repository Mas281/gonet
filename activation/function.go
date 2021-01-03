package activation

import "math"

var Sigmoid = Function{sigmoid, sigmoidPrime}

var sigmoid = func(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

var sigmoidPrime = func(x float64) float64 {
	sig := sigmoid(x)
	return sig * (1 - sig)
}

type Function struct {
	Func       func(float64) float64
	Derivative func(float64) float64
}
