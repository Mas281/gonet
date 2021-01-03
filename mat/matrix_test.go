package mat

import (
	"math/rand"
	"testing"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func BenchmarkMultiply(b *testing.B) {
	x := NewEmpty(100, 100)
	x.Apply(func(float64) float64 {
		return rand.Float64()
	})

	y := NewEmpty(100, 100)
	y.Apply(func(float64) float64 {
		return rand.Float64()
	})

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		Dot(x, y)
	}
}

func BenchmarkTranspose(b *testing.B) {
	m := NewEmpty(100, 100)
	m.Apply(func(float64) float64 {
		return rand.Float64()
	})

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		Transpose(m)
	}
}
