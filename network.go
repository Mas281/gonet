package network

import (
	"bufio"
	"encoding/json"
	"errors"
	"github.com/Mas281/gonet/activation"
	"github.com/Mas281/gonet/mat"
	"github.com/Mas281/gonet/training"
	"github.com/Mas281/gonet/uniform"
	"io/ioutil"
	"math/rand"
	"os"
	"time"
)

type Network struct {
	InputSize int
	Layers    int

	Weights    []mat.Matrix

	LearningRate float64
}

var errLayers = errors.New("network must have at least 3 layers")

func New(learningRate float64, layerSizes ...int) *Network {
	if len(layerSizes) < 3 {
		panic(errLayers)
	}

	rand.Seed(time.Now().UnixNano())

	inputSize := layerSizes[0]
	weights := make([]mat.Matrix, len(layerSizes)-1)

	for i := 1; i < len(layerSizes); i++ {
		size := layerSizes[i]
		prevSize := layerSizes[i-1]

		distribution := uniform.Make(prevSize)

		matrix := mat.NewEmpty(size, prevSize)
		matrix.Apply(func(float64) float64 {
			return distribution.Generate()
		})

		weights[i-1] = matrix
	}

	return &Network{
		inputSize,
		len(layerSizes),
		weights,
		learningRate,
	}
}

var errInputsSize = errors.New("wrong number of inputs")

func (net Network) Predict(input []float64) mat.Matrix {
	outputs := net.layerOutputs(input)
	return outputs[len(outputs)-1]
}

func (net *Network) Train(data training.Data) {
	outputs := net.layerOutputs(data.Input)

	lastLayer := true
	var lastErrors mat.Matrix

	deltas := make([]mat.Matrix, len(outputs)-1)

	for i := len(outputs) - 1; i > 0; i-- {
		// dE/do
		var dErrorWrtOutput mat.Matrix

		if lastLayer {
			// dE/do = -(t - o) = o - t
			dErrorWrtOutput = mat.Subtract(outputs[i], data.TargetMatrix())
			lastLayer = false
		} else {
			dErrorWrtOutput = mat.Dot(mat.Transpose(net.Weights[i]), lastErrors)
		}

		lastErrors = dErrorWrtOutput

		// do/dΣ = o(1 - o)
		dOutputWrtSum := mat.Copy(outputs[i])
		dOutputWrtSum.Apply(func(value float64) float64 {
			return value * (1.0 - value)
		})

		// dΣ/dw = o(previous)
		dSumWrtWeights := outputs[i-1]

		// Delta
		deltaWeights := mat.Dot(mat.MulElem(dErrorWrtOutput, dOutputWrtSum), mat.Transpose(dSumWrtWeights))
		deltaWeights.Scale(net.LearningRate)

		deltas[i-1] = deltaWeights
	}

	for i, delta := range deltas {
		net.Weights[i] = mat.Subtract(net.Weights[i], delta)
	}
}

func (net Network) layerOutputs(input []float64) []mat.Matrix {
	if len(input) != net.InputSize {
		panic(errInputsSize)
	}

	layerOutput := mat.NewColumn(input)
	outputs := []mat.Matrix{layerOutput}

	for _, weights := range net.Weights {
		layerOutput = mat.Dot(weights, layerOutput)
		layerOutput.Apply(activation.Sigmoid.Func)

		outputs = append(outputs, layerOutput)
	}

	return outputs
}

func (net Network) Save(fileName string) error {
	file, err := os.Create(fileName)

	if err != nil {
		return err
	}

	defer func() {
		_ = file.Close()
	}()

	writer := bufio.NewWriter(file)
	marshalled, err := json.MarshalIndent(net, "", "    ")

	if err != nil {
		return err
	}

	_, err = writer.Write(marshalled)

	if err != nil {
		return err
	}

	err = writer.Flush()
	return err
}

func Load(fileName string) (*Network, error) {
	file, err := os.Open(fileName)

	if err != nil {
		return nil, err
	}

	defer func() {
		_ = file.Close()
	}()

	reader := bufio.NewReader(file)
	bytes, err := ioutil.ReadAll(reader)

	if err != nil {
		return nil, err
	}

	var network Network
	err = json.Unmarshal(bytes, &network)

	return &network, err
}
