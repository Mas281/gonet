package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"fmt"
	"github.com/Mas281/gonet"
	"github.com/Mas281/gonet/mat"
	"github.com/Mas281/gonet/training"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	net := parseNetwork()

	if net == nil {
		return
	}

	fmt.Println("===== Testing =====")
	rand.Seed(time.Now().UnixNano())
	test(net, false)
}

func train(epochs int) *network.Network {
	fmt.Println("==== Training ===== ")
	start := time.Now()
	net := network.New(0.1, 784, 200, 10)
	fmt.Println("Training with", epochs, "epochs")

	datas := csvToDatas("mnist_train.csv")

	for i := 0; i < epochs; i++ {
		for _, data := range datas {
			net.Train(data)
		}
		fmt.Println("Completed epoch", strconv.Itoa(i+1)+"/"+strconv.Itoa(epochs))
	}

	fmt.Println("Completed training in", time.Since(start))
	return net
}

func test(net *network.Network, showImages bool) {
	start := time.Now()
	datas := csvToDatas("mnist_test.csv")

	numCorrect := 0

	for _, data := range datas {
		output := net.Predict(data.Input)

		prediction := highestFrom(output)
		actual := highestFrom(data.TargetMatrix())
		correct := prediction == actual

		if showImages && (rand.Intn(50) == 0 || (!correct && rand.Intn(2) == 0)) {
			saveImage(data.Input)

			fmt.Println("Prediction:", prediction)
			fmt.Println("Actual:", actual)

			input := bufio.NewScanner(os.Stdin)
			input.Scan()
		}

		if correct {
			numCorrect++
		}
	}

	if !showImages {
		fmt.Println("Completed testing in", time.Since(start))
		fmt.Println(numCorrect, "correct out of", len(datas), "training samples")
		fmt.Println("Accuracy:", (float64(numCorrect)/float64(len(datas)))*100)
	}
}

func saveImage(pixels []float64) {
	img := image.NewGray(image.Rectangle{
		Min: image.Point{},
		Max: image.Point{X: 28, Y: 28},
	})

	row := 0
	column := 0

	for _, pixel := range pixels {
		if column == 28 {
			row++
			column = 0
		}

		colour := uint8(math.Round(pixel * 255))
		img.SetGray(column, row, color.Gray{Y: colour})

		column++
	}

	var buf bytes.Buffer
	err := png.Encode(&buf, img)

	if err != nil {
		panic(err)
	}

	file, _ := os.Create("image.png")
	defer file.Close()

	err = png.Encode(file, img)

	if err != nil {
		panic(err)
	}
}

func highestFrom(output mat.Matrix) int {
	highest := 0.0
	highestNo := 0

	for i := 0; i < 10; i++ {
		chance := output.Get(0, i)

		if chance > highest {
			highest = chance
			highestNo = i
		}
	}

	return highestNo
}

func csvToDatas(fileName string) []training.Data {
	file, _ := os.Open(fileName)
	defer func() {
		_ = file.Close()
	}()

	reader := csv.NewReader(file)
	var datas []training.Data

	for {
		record, err := reader.Read()

		if err == io.EOF {
			break
		}

		datas = append(datas, recordToData(record))
	}

	return datas
}

func recordToData(record []string) training.Data {
	input := make([]float64, 784)

	for i := 1; i < len(record); i++ {
		f, _ := strconv.ParseFloat(record[i], 64)
		input[i-1] = ((f / 255) * 0.99) + 0.01
	}

	target := make([]float64, 10)

	for i := range target {
		target[i] = 0.01
	}

	actual, _ := strconv.Atoi(record[0])
	target[actual] = 0.99

	return training.Data{
		Input:  input,
		Target: target,
	}
}

func parseNetwork() *network.Network {
	scanner := bufio.NewScanner(os.Stdin)

	if scanner.Scan() {
		args := strings.Split(scanner.Text(), " ")

		if len(args) != 2 {
			fmt.Println("Invalid arguments")
			return nil
		}

		if strings.EqualFold(args[0], "train") {
			epochs, err := strconv.Atoi(args[1])

			if err != nil {
				fmt.Println("Invalid number of epochs")
				return nil
			}

			net := train(epochs)

			fileName := "trained_" + strconv.Itoa(epochs) + ".gonet"
			fmt.Println("Savings as", fileName)

			err = net.Save(fileName)

			if err != nil {
				fmt.Println("Error while saving network")
				panic(err)
			}

			return net
		} else if strings.EqualFold(args[0], "load") {
			fileName := args[1]
			net, err := network.Load(fileName)

			if err != nil {
				fmt.Println("Error while loading network")
				panic(err)
			}

			fmt.Println("Loaded network", fileName)
			return net
		}
	}

	return nil
}
