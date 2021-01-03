package uniform

import (
	"math"
	"math/rand"
)

type Uniform struct {
	Min float64
	Max float64
}

func Make(layerSize int) Uniform {
	sqrtLs := math.Sqrt(float64(layerSize))

	return Uniform{
		Min: -1 / sqrtLs,
		Max: 1 / sqrtLs,
	}
}

func (u Uniform) Generate() float64 {
	return u.Min + (rand.Float64() * (u.Max - u.Min))
}
